import numpy as np
import pandas as pd
from scipy.stats import norm
from math import *
from numpy.linalg import det, cholesky, multi_dot, cond
from scipy.linalg import cho_solve, cho_factor, solve, pinv
import re
from functools import partial
import sys
import itertools
import warnings
from warnings import simplefilter
import os
from concurrent.futures import ProcessPoolExecutor
import streamlit as st

st.set_page_config(layout='wide', page_title="3M Korea",page_icon="🧊")

global nobs, nchocc, nvarma, nc,ivgenva_rum, nCholErr, Parametrized
global upper_limit, nind, seed, seed1, Non_IID, D_matrix, _ranper, GHK, Data_Split, Approx_GHK, Est_Normal, Non_comp, Logit, Probit, IID_Err, Err_Full
global req_col, altchm, MACMLS, ncholomega, mix_ele, mu_ordering, Choice_Col, Treat_CS

row_wise = 1
col_wise = 0

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def split_brand_string(s):
	"""
	Extracts the value inside parentheses and cleans the value outside the parentheses.

	Parameters:
	- s (str): Input string like 'Brand_1 (Scotch-Brite by 3M)'

	Returns:
	- tuple: (outside_cleaned, inside_parentheses)
	"""
	# Extract inside parentheses
	match_inside = re.search(r'\((.*?)\)', s)
	inside = match_inside.group(1) if match_inside else None

	# Extract part before the parentheses
	outside_raw = s.split('(')[0].strip()

	# Remove _number suffix using regex
	outside_cleaned = re.sub(r'_\d+$', '', outside_raw)

	return outside_cleaned, inside

# Procedure to print dataframe ina nice tabular format
def pprint_df(dframe):
	print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


# Procedure to generate Halton Draws
def HaltonSequence(n, dim):
	prim = np.array(
		[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
		 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
		 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
		 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
		 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541])
	prim = prim[:, np.newaxis]
	hs = np.zeros((n, dim))
	for idim in range(dim):
		b = prim[idim, 0]
		hs[:, idim] = halton(n, b)

	return (hs[10:n, :])


# Halton draws sub-procedure. This is where we braek the unit length stick infinite times
def halton(n, s):
	k = floor(log(n + 1) / log(s))
	phi = np.zeros((1, 1))
	i = 1
	count = 0
	while i <= k:
		count = count + 1
		x = phi
		j = 1
		while j < s:
			y = phi + (j / s ** i)
			x = np.vstack((x, y))
			j = j + 1

		phi = x
		i = i + 1

	x = phi
	j = 1
	while (j < s) and (len(x) < (n + 1)):
		y = phi + (j / s ** i)
		x = np.vstack((x, y))
		j = j + 1

	out = x[1:(n + 1), 0]
	return (out)


# GHK simulator for calculation of multivariate Normal-distribution CDF
def cdfmvnGHK(a, r, s):
	global _halt_maxdraws, _halt_numdraws, allHaltDraws, nrep
	a = np.multiply(a, (a < 5.7)) + 5.7 * (a >= 5.7)
	a = np.multiply(a, (a > -5.7)) - 5.7 * (a <= -5.7)

	nintegdim = a.shape[1]
	if sys.getsizeof(s) < 50:
		np.random.seed(s)
	else:
		np.random.set_state(s)

	rnum = np.random.random((1, 1))
	rnum = rnum[0, 0]
	ss = np.random.get_state()
	s = ss

	startRow = ceil(rnum * (_halt_maxdraws - _halt_numdraws - 1))
	uniRands = allHaltDraws[startRow:startRow + _halt_numdraws, 0:nintegdim - 1]
	chol_r = cholesky(r)
	chol_r = chol_r.T
	ghkArr = np.zeros((nrep, nintegdim))
	etaArr = np.zeros((nrep, (nintegdim - 1)))
	temp = norm.cdf(a[0, 0] / chol_r[0, 0]) * np.ones((nrep, 1))
	ghkArr[:, 0] = temp[:, 0]
	del temp

	for iintegdim_main in range(1, nintegdim, 1):
		iintegdim = iintegdim_main - 1
		temp1 = uniRands[:, iintegdim]
		temp2 = ghkArr[:, iintegdim]
		temp1 = temp1[:, np.newaxis]
		temp2 = temp2[:, np.newaxis]

		temp3 = np.multiply(temp1, temp2)
		temp4 = cdfni(temp3)
		temp4 = temp4[:, np.newaxis]
		etaArr[:, iintegdim] = temp4[:, 0]
		del temp1, temp2, temp3, temp4

		ghkElem = a[0, iintegdim + 1] * np.ones((nrep, 1))
		ghkElem1 = 0
		for jintegdim in range(0, iintegdim_main, 1):
			temp = chol_r[jintegdim, iintegdim + 1] * etaArr[:, jintegdim]
			temp = temp[:, np.newaxis]
			ghkElem1 = ghkElem1 - temp
			del temp

		ghkElem1 = ghkElem1 + ghkElem
		temp1 = ghkElem1 / (chol_r[(iintegdim + 1), (iintegdim + 1)])
		temp2 = cdfn(temp1)
		temp2 = temp2[:, np.newaxis]
		ghkArr[:, iintegdim + 1] = temp2[:, 0]
		del temp1, temp2

	probab = ghkArr[:, 0]
	probab = probab[:, np.newaxis]
	for iintegdim in range(1, nintegdim, 1):
		temp = ghkArr[:, iintegdim]
		temp = temp[:, np.newaxis]
		probab = np.multiply(probab, temp)
		del temp

	probab = np.mean(probab, axis=0)[0]
	return (probab, s)


# Procedure to obtain standard normal distribution CDF
def cdfn(a):
	out = norm.cdf(a[:, 0])
	return (out)


# Procedure to obtain inverse of univariate normal distribution CDF
def cdfni(a):
	out = norm.ppf(a[:, 0])
	return (out)


def pdfnd(x):
	p1 = exp(-0.5 * np.multiply(x, x))
	p2 = sqrt(2 * pi)
	p = p1 / p2
	store = -np.multiply(p, x)
	return (store)


def pdfn(a):
	out = norm.pdf(a[:, 0])
	return (out)


# Procedure to convert co-variance matrix into correlation matrix
def corrvc(S):
	temp = np.diag(S)
	temp = temp[:, np.newaxis]
	D = temp ** 0.5
	Dcol = np.kron(np.ones((1, S.shape[0])), D)
	Drow = np.kron(np.ones((S.shape[0], 1)), D.T)
	DF = np.divide(S, Dcol)
	DF = np.divide(DF, Drow)
	R = diagrv(DF)
	return (R)


# Procedure to obtain Cholesky decomposition
def chol(r):
	a = cholesky(r)
	return (a)


# Procedure to put 1 on diagonal of a matrix
def diagrv(a):
	for i in range(0, a.shape[0], 1):
		a[i, i] = 1.0
	return (a)


# Procedure to put the vector on the diagonal of a matrix
def diagput(a, b):
	for i in range(0, a.shape[0], 1):
		a[i, i] = b[i]
	return (a)


# Procedure to check positive-definiteness of a matrix
def pd_inv1(a):
	n = a.shape[0]
	I = np.identity(n)
	check1 = np.isfinite(cond(a))
	det1 = det(a)
	check2 = det1 > 0.01
	check = check1 & check2
	if (check):
		return cho_solve(cho_factor(a, lower=True), I)
	else:
		return (pinv(a))


# Procedure to check positive-definiteness of a matrix
def pd_inv(a):
	n = a.shape[0]
	I = np.identity(n)
	check1 = np.isfinite(cond(a))
	det1 = det(a)
	check2 = det1 > 0.01
	check3 = is_pos_def(a)
	check = check1 & check2 & check3
	if (check):
		return solve(a, I, sym_pos=True, overwrite_b=True)
	else:
		return (pinv(a))


# Procedure to check positive-definiteness of a matrix
def is_pos_def(x):
	return np.all(np.linalg.eigvals(x) > 0)


# Procedure to expand a vector into symmetric matrix
def xpnd(r):
	d = int((-1 + sqrt(1 + 8 * len(r))) / 2)
	xp = np.zeros((d, d), dtype=float)
	count = 0
	xp[0, 0] = r[0, 0]
	for i in range(1, d, 1):
		for j in range(0, i + 1, 1):
			count = count + 1
			xp[i, j] = r[count, 0]
			xp[j, i] = r[count, 0]
	return (xp)


# Procedure to vectorize a symmetric matrix
def vech(r):
	drow = r.shape[0]
	d = int(drow * (drow + 1) * 0.5)
	xp = np.zeros((d, 1))
	xp[0, 0] = r[0, 0]
	count = 0
	for i in range(1, drow, 1):
		for j in range(0, i + 1, 1):
			count = count + 1
			xp[count, 0] = r[i, j]
	return (xp)


# Procedure to extract upper triangular matrix
def upmat(r):
	drow = r.shape[0]
	xp = np.zeros((drow, drow))
	for i in range(0, drow, 1):
		for j in range(i, drow, 1):
			xp[i, j] = r[i, j]
	return (xp)


# Procedure to extract lower triangular matrix
def lowmat(r):
	drow = r.shape[0]
	xp = np.zeros((drow, drow))
	xp[0, 0] = r[0, 0]
	for i in range(1, drow, 1):
		for j in range(0, i + 1, 1):
			xp[i, j] = r[i, j]
	return (xp)


def divide_into_bins(total: int, num_bins: int):
	if total <= 0 or num_bins <= 0:
		raise ValueError("Both total and num_bins must be positive integers")
	if num_bins > total:
		raise ValueError("Number of bins cannot exceed total")

	base_size = total // num_bins
	remainder = total % num_bins

	bins = []
	start = 1

	for i in range(1, num_bins + 1):
		# Distribute remainder across the first few bins
		extra = 1 if i <= remainder else 0
		end = start + base_size + extra - 1
		bins.append((start, end))
		start = end + 1

	return bins

# Multi-threaded Likelihood Calling procedure
def lpr(x,Data,dp_progress):
	if Num_Threads > 1:
		data_list = [iter for iter in range(0, Num_Threads, 1)]
		with ProcessPoolExecutor(max_workers=Num_Threads) as executor:
			prod_x = partial(lprT, parm=x)
			result_list = list(executor.map(prod_x, data_list))

		a_temp = list(itertools.chain.from_iterable(result_list))
		atemp_array = np.asarray(a_temp)
	else:
		atemp_array = lprT(0, x, Data, dp_progress)


	LL_Value = atemp_array
	return (LL_Value)

# Likelihood procedure
def lprT(iter, parm, Data, dp_progress):
	message_formatted1 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Calulating utilities</p>'
	message_formatted2 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Calculating Probability</p>'


	dp_progress.markdown(message_formatted1, unsafe_allow_html=True)
	if (iter == 0 and iter + 1 == Num_Threads):
		st_iter = int(0)
		end_iter = int(nobs - 1)
	elif (iter == 0 and iter + 1 != Num_Threads):
		st_iter = int(Data_Split[iter, 0]) * nchocc
		end_iter = int(Data_Split[iter, 1]) * nchocc
	else:
		st_iter = int(Data_Split[iter, 0] - 1) * nchocc + 1
		if (iter + 1 < Num_Threads):
			end_iter = int(Data_Split[iter, 1]) * nchocc
		else:
			end_iter = int(nobs - 1)
	nobs_num = int(end_iter - st_iter + 1)
	nind_num = (int(Data_Split[iter, 1] - Data_Split[iter, 0] + 1))


	smallb = parm[0:nvarma]
	if Model_Type	== 'MNP':
		temp = parm[nvarma:nvarma + nCholErr]
		Psi1 = xpnd(temp)
		del temp
		Psi = np.zeros((nc, nc))
		Psi[1:nc, 1:nc] = Psi1

	Param_Expand = (np.kron(np.ones((nc, 1)), smallb))
	v1 =  Param_Expand * (Data.loc[st_iter:end_iter, ivgenva_rum].values.T)

	Utility = np.empty(shape=(nobs_num, nc), dtype=float)
	for i in range(0, nc, 1):
		j = i + 1
		Utility[:, i] = np.sum(v1[(j - 1) * nvarma:(j * nvarma), :], axis=0)
	del v1


	dp_progress.markdown(message_formatted2, unsafe_allow_html=True)
	pbar = st.empty()
	if Model_Type == 'MNP':
		Likelihood = np.zeros((nind_num, nc))
		seednext = MACMLS[iter]
		iden_matrix = np.eye(nc - 1)
		one_negative = -1 * np.ones((nc - 1, 1))
		for i in range(0, nobs_num, 1):
			variable_output = f'Share Calculation Progress : {round(((i + 1) / nobs_num) * 100)}%'
			html_str = f"""<style>p.a{{font-size:26px;font-family:sans serif;color:blue; text-align:center}}</style><p class="a">{variable_output}</p>"""
			dp_progress.markdown(html_str, unsafe_allow_html=True)
			Full_error = Psi

			Utility_curr = Utility[i, :]
			try:
				dim2 = Utility_curr.shape[1]
			except:
				Utility_curr = Utility_curr[:, np.newaxis]

			for Alt_chosen in range(1,nc+1):
				if (Alt_chosen == 1):
					temp1 = np.hstack((one_negative, iden_matrix))
				elif (Alt_chosen == nc):
					temp1 = np.hstack((iden_matrix, one_negative))
				else:
					ch = int(Alt_chosen)
					t1 = iden_matrix[:, 0:ch - 1]
					t2 = iden_matrix[:, ch - 1:nc - 1]
					temp1 = np.hstack((t1, one_negative, t2))

				M_big = temp1
				Mean_changed = multi_dot([M_big, Utility_curr])
				Error_changed = multi_dot([M_big, Full_error, M_big.T])

				mean_gu = -Mean_changed
				var_gu = Error_changed

				om = np.diag(var_gu)
				om = om[:, np.newaxis]
				om = om ** 0.5
				mean_final = np.divide(mean_gu, om)

				var_final = corrvc(var_gu)
				nc_curr = var_final.shape[0]				
				seed20 = seednext
				p4_temp, sss = cdfmvnGHK(mean_final.T, var_final, seed20)
				seednext = sss			

				Likelihood[i, Alt_chosen-1] = p4_temp

		output = Likelihood
	else:
		Utility_exp = np.exp(Utility)
		Utility_sum = np.sum(Utility_exp, axis=1)
		Prob = Utility_exp / Utility_sum[:, np.newaxis]
		output = Prob

	html_str = f"""<style>p.a{{font-size:26px;font-family:sans serif;color:blue; text-align:center}}</style><p class="a"></p>"""
	dp_progress.markdown(html_str, unsafe_allow_html=True)
	return output


# Procedure to construct full beta vector from various fixed and active parameter sub-vectors
def reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
	total_parm = parm_estimated.shape[0] + parm_fixed.shape[0]
	beta_reconstructed = np.empty(shape=(total_parm, 1), dtype=float)

	beta_reconstructed[idx_estimated] = parm_estimated
	beta_reconstructed[idx_fixed] = parm_fixed
	return (beta_reconstructed)


def Calculate_Share(Scenario_df,Mapping_df, dp_progress):
	global MACMLS,nrephalt,Data_Split,Num_Threads,nobs,nvarma,Model_Type,nCholErr,nc,ivgenva_rum,_halt_maxdraws,_halt_numdraws,allHaltDraws,nrep

	Model_Type	             = 'MNP'  # Set to 'MNP' for Mixed Logit, 'MNL' for Multinomial Logit, 'Probit' for Probit model
	Model_Config             = 'Labeled'  # Set to 'Labeled' for labeled data, 'Unlabeled' for unlabeled data

	encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'utf-16', 'iso-8859-1']
	current_dir = os.path.dirname(os.path.abspath(__file__))
	message_formatted1 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Reading Population file</p>'
	message_formatted2 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Reading Parameters file</p>'
	dp_progress.markdown(message_formatted1, unsafe_allow_html=True)

	if Model_Config   == 'Labeled':
		file_name = os.path.join(current_dir, f'3M_Korea_Simulator_Data.xlsx')
	else:
		file_name = os.path.join(current_dir, f'3M_Korea_MNL_Estimation_Data.csv')

	Main_data = pd.read_excel(file_name,sheet_name='Data')

	dp_progress.markdown(message_formatted2, unsafe_allow_html=True)

	if Model_Config == 'Labeled':
		Parameter_filename = os.path.join(current_dir, f'Korea_Parameters_{Model_Type}_AS.csv')
	else:
		Parameter_filename = os.path.join(current_dir, f'Korea_Parameters_{Model_Type}.csv')
	for enc in encodings_to_try:
		try:
			External_Param = pd.read_csv(Parameter_filename, encoding=enc)
			break
		except UnicodeDecodeError:
			continue

	if External_Param.shape[1] == 2:
		External_Param.columns = ['Parameter', 'Value']  # Renaming columns to 'Parameter' and 'Value'
	elif External_Param.shape[1] == 5:
		External_Param.columns = ['Parameter', 'Value', 'St-Err', 'P-Value', 'T-Value']  # Renaming columns to 'Parameter', 'Value', 'St-Err', 'P-Value', 'T-Value'
	elif External_Param.shape[1] == 7:
		External_Param.columns = ['Alternative','Attribute','Parameter', 'Value', 'St-Err', 'P-Value', 'T-Value']  # Renaming columns to 'Parameter', 'Value', 'St-Err', 'P-Value', 'T-Value'

	External_Param = External_Param['Value'].values
	try:
		dim2 = External_Param.shape[1]
	except:
		External_Param = External_Param[:, np.newaxis]

	nc       = 12  # Number of alternatives in the choice set
	nc_shown = 3  # Number of alternatives shown in the choice set

	message_formatted3 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Preparing Data</p>'
	dp_progress.markdown(message_formatted3, unsafe_allow_html=True)

	Scenario_df_matrix = Scenario_df.values
	for i in range(Scenario_df_matrix.shape[0]):
		curr_att = Scenario_df_matrix[i, 0]
		for j in range(1,Scenario_df_matrix.shape[1]):
			curr_level = Scenario_df_matrix[i, j]
			curr_att_nc = curr_att + '_' + str(j)
			Main_data[curr_att_nc] = curr_level


	Mapping_df_columns = Mapping_df.columns.tolist()
	for map_col in Mapping_df_columns:
		map_col_values = Mapping_df[map_col].values
		map_col_values = map_col_values[~pd.isnull(map_col_values)]
		for i in range(nc):
			curr_col = f'{map_col}_{i + 1}'
			for map_value in map_col_values:
				new_col_name = f"{curr_col} ({map_value})"
				Main_data[new_col_name] = (Main_data[curr_col]==map_value).astype(int)



	Price_col = [f'Price_{i + 1}' for i in range(nc)]
	Main_data[Price_col] = Main_data[Price_col] / 100
	Main_data.loc[:, Price_col] = Main_data.loc[:, Price_col].replace(0, 1)
	Main_data.loc[:, Price_col] = (Main_data.loc[:, Price_col]).apply(np.log)
	Main_data.loc[:, Price_col] = 1 * Main_data.loc[:, Price_col]



	altid = ['New_ID']
	Main_data['New_ID'] = range(1, Main_data.shape[0] + 1)
	Main_data['New_ID'] = Main_data['New_ID'].astype(int)
	Main_data['sero'] = 0
	Main_data['uno'] = 1

	Pool_Based_MP   = 0  # Set to 1 if using Pool based multiprocessing, 0 if using Process based multiprocessing
	Num_Threads     = 1  # cpu_count()       # Number of threads to be used for multithreading based on number of cpu cores


	if Model_Type != 'MNP':
		Num_Threads = 1


	nind = int(Main_data.shape[0])  # Number of individuals in the sample
	nchocc = 1  # Number of choice occassions per indivudal
	nobs = int(nind * nchocc)  # Sample size

	upper_limit = 1e-05  # Any value of CDF below this limit is considered as zero

	# ---------------------------------------------------------
	# DO not touch this section. Setting estimation related vriables
	# --------------------------------------------------------
	Num_MNP = int((nc - 1) > 2)
	if Model_Type == 'MNP':
		message_formatted4 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Generating Halton Draws</p>'
		dp_progress.markdown(message_formatted4, unsafe_allow_html=True)
		if (Num_MNP > 0):
			ncol = (nc - 1)
			_halt_numdraws = 50
			nrep = _halt_numdraws
			nrephalt = nrep
			allHaltDraws = HaltonSequence(nobs * (nrep + 10), int(ncol))
			_halt_maxdraws = allHaltDraws.shape[0]

		MACMLS = [300000 + i for i in range(Num_Threads)]

	if Model_Config == 'Labeled':
		Alt_labels = ['3-layers/2-sided scrub sponge', 'Acrylic scourer', 'Bottle cleaner with handle', 'Disposable Sheet',
					  'Handled Dishwand', 'Large sheet/wipe', 'Metal Ball Scourer', 'Net Sponge', 'Net/Mesh cloth ',
					  'Scrub Pad', 'Scrub Sponge', 'Sponge']

		All_UT_Spec = [['Brand_1 (Byulpyo)', 'Brand_1 (Cleanwrap)', 'Brand_1 (Daiso)', 'Brand_1 (Frog)', 'Brand_1 (No Brand)', 'Brand_1 (Scotch-Brite by 3M)', 'Brand_1 (Scott (Yuhan-Kimberly))', 'Brand_1 (Spontex)', 'Cleaning strength_1 (Extra tough/ extra heavy duty)', 'Cleaning strength_1 (Gentle/ delicate cleaning)', 'Cleaning strength_1 (Tough Cleaning/ Heavy duty )', 'Surface types_1 (Cast Iron (Cookware/Grills/etc))', 'Surface types_1 (Everyday dishware & utensils)', 'Surface types_1 (Glass (Cooktop))', 'Surface types_1 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_1 (Plastic (Dishware, Utensils, etc))', 'Surface types_1 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_1 (Stainless Steel (Cooktop))', 'Surface types_1 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_1 (Stainless Steel (Sink /Cookware) )', 'Surface types_1 (Suitable for use across most surface types)', 'Stain types_1 (Effective at removing different stuck on and oily stains)', 'Stain types_1 (Good at removing heavy oil stains)', 'Stain types_1 (Good at removing light (non-oily) stains)', 'Stain types_1 (Good at removing light oil stains)', 'Stain types_1 (Good at removing sticky stains)', 'Stain types_1 (Good at removing stubborn burnt stains)', 'Stain types_1 (Good at removing tea/coffee stains)', 'Benefit_1 (Antibacterial/Prevent bacterial growth)', 'Benefit_1 (Antimicrobial/Prevent mold & odor)', 'Benefit_1 (Durable/long lasting)', 'Benefit_1 (Easy to grip)', 'Benefit_1 (Easy to reach and clean tight corners and grooves)', "Benefit_1 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_1 (Foam well with a small amount of detergent)', 'Benefit_1 (Stain resistant)', 'Benefit_1 (Will not scratch or damage cleaning surfaces)', 'Claim_1 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_1 (Create a shiny finish after use)', 'Claim_1 (Eco-friendly/Good for the environment)', 'Claim_1 (Fun designs/ shapes to elevate your mood)', 'Claim_1 (Guarantee satisfaction after use)', 'Claim_1 (Keeps your family safe)', 'Claim_1 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_1 (Scented (emits pleasant scent while cleaning))', 'Claim_1 (Transform cleaning into a fun experience with foamy bubble)', 'Material_1 (100% Recycled PET plastic)', 'Material_1 (Flexible foam (texture changes with water temperature))', 'Material_1 (Man-made fibre (e.g., polyester))', 'Material_1 (Melamine foam (magic eraser))', 'Material_1 (Natural cellulose sponge)', 'Material_1 (Natural coconut based fibre)', 'Material_1 (Natural corn based fibre)', 'Material_1 (Stay Fresh foam (foam that resist stains and odors))', 'Price_1', 'Color_1 (Beige/Brown )', 'Color_1 (Black )', 'Color_1 (Blue)', 'Color_1 (Dark Green)', 'Color_1 (Green)', 'Color_1 (Orange)', 'Color_1 (Pink)', 'Color_1 (Purple)', 'Color_1 (White)', 'Color_1 (Yellow)', 'Shape_1 (Flower)', 'Shape_1 (Leaf)', 'Shape_1 (Oval )', 'Shape_1 (Rectangle)', 'Shape_1 (Round)', 'Shape_1 (Smiley)', 'Shape_1 (Square)', 'Shape_1 (Tear drop )', 'Shape_1 (Wave )'],
	['Brand_2 (Byulpyo)', 'Brand_2 (Cleanwrap)', 'Brand_2 (Daiso)', 'Brand_2 (Frog)', 'Brand_2 (No Brand)', 'Brand_2 (Scotch-Brite by 3M)', 'Brand_2 (Scott (Yuhan-Kimberly))', 'Brand_2 (Spontex)', 'Cleaning strength_2 (Extra tough/ extra heavy duty)', 'Cleaning strength_2 (Gentle/ delicate cleaning)', 'Cleaning strength_2 (Tough Cleaning/ Heavy duty )', 'Surface types_2 (Cast Iron (Cookware/Grills/etc))', 'Surface types_2 (Everyday dishware & utensils)', 'Surface types_2 (Glass (Cooktop))', 'Surface types_2 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_2 (Plastic (Dishware, Utensils, etc))', 'Surface types_2 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_2 (Stainless Steel (Cooktop))', 'Surface types_2 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_2 (Stainless Steel (Sink /Cookware) )', 'Surface types_2 (Suitable for use across most surface types)', 'Stain types_2 (Effective at removing different stuck on and oily stains)', 'Stain types_2 (Good at removing heavy oil stains)', 'Stain types_2 (Good at removing light (non-oily) stains)', 'Stain types_2 (Good at removing light oil stains)', 'Stain types_2 (Good at removing sticky stains)', 'Stain types_2 (Good at removing stubborn burnt stains)', 'Stain types_2 (Good at removing tea/coffee stains)', 'Benefit_2 (Antibacterial/Prevent bacterial growth)', 'Benefit_2 (Antimicrobial/Prevent mold & odor)', 'Benefit_2 (Durable/long lasting)', 'Benefit_2 (Easy to grip)', 'Benefit_2 (Easy to reach and clean tight corners and grooves)', "Benefit_2 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_2 (Foam well with a small amount of detergent)', 'Benefit_2 (Stain resistant)', 'Benefit_2 (Will not scratch or damage cleaning surfaces)', 'Claim_2 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_2 (Create a shiny finish after use)', 'Claim_2 (Eco-friendly/Good for the environment)', 'Claim_2 (Fun designs/ shapes to elevate your mood)', 'Claim_2 (Guarantee satisfaction after use)', 'Claim_2 (Keeps your family safe)', 'Claim_2 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_2 (Scented (emits pleasant scent while cleaning))', 'Claim_2 (Transform cleaning into a fun experience with foamy bubble)', 'Material_2 (100% Recycled PET plastic)', 'Material_2 (Flexible foam (texture changes with water temperature))', 'Material_2 (Man-made fibre (e.g., polyester))', 'Material_2 (Melamine foam (magic eraser))', 'Material_2 (Natural cellulose sponge)', 'Material_2 (Natural coconut based fibre)', 'Material_2 (Natural corn based fibre)', 'Material_2 (Stay Fresh foam (foam that resist stains and odors))', 'Price_2', 'Color_2 (Beige/Brown )', 'Color_2 (Black )', 'Color_2 (Blue)', 'Color_2 (Dark Green)', 'Color_2 (Green)', 'Color_2 (Orange)', 'Color_2 (Pink)', 'Color_2 (Purple)', 'Color_2 (White)', 'Color_2 (Yellow)', 'Shape_2 (Flower)', 'Shape_2 (Leaf)', 'Shape_2 (Oval )', 'Shape_2 (Rectangle)', 'Shape_2 (Round)', 'Shape_2 (Smiley)', 'Shape_2 (Square)', 'Shape_2 (Tear drop )', 'Shape_2 (Wave )'],
	['Brand_3 (Byulpyo)', 'Brand_3 (Cleanwrap)', 'Brand_3 (Daiso)', 'Brand_3 (Frog)', 'Brand_3 (No Brand)', 'Brand_3 (Scotch-Brite by 3M)', 'Brand_3 (Scott (Yuhan-Kimberly))', 'Brand_3 (Spontex)', 'Cleaning strength_3 (Extra tough/ extra heavy duty)', 'Cleaning strength_3 (Gentle/ delicate cleaning)', 'Cleaning strength_3 (Tough Cleaning/ Heavy duty )', 'Surface types_3 (Cast Iron (Cookware/Grills/etc))', 'Surface types_3 (Everyday dishware & utensils)', 'Surface types_3 (Glass (Cooktop))', 'Surface types_3 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_3 (Plastic (Dishware, Utensils, etc))', 'Surface types_3 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_3 (Stainless Steel (Cooktop))', 'Surface types_3 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_3 (Stainless Steel (Sink /Cookware) )', 'Surface types_3 (Suitable for use across most surface types)', 'Stain types_3 (Effective at removing different stuck on and oily stains)', 'Stain types_3 (Good at removing heavy oil stains)', 'Stain types_3 (Good at removing light (non-oily) stains)', 'Stain types_3 (Good at removing light oil stains)', 'Stain types_3 (Good at removing sticky stains)', 'Stain types_3 (Good at removing stubborn burnt stains)', 'Stain types_3 (Good at removing tea/coffee stains)', 'Benefit_3 (Antibacterial/Prevent bacterial growth)', 'Benefit_3 (Antimicrobial/Prevent mold & odor)', 'Benefit_3 (Durable/long lasting)', 'Benefit_3 (Easy to grip)', 'Benefit_3 (Easy to reach and clean tight corners and grooves)', "Benefit_3 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_3 (Foam well with a small amount of detergent)', 'Benefit_3 (Stain resistant)', 'Benefit_3 (Will not scratch or damage cleaning surfaces)', 'Claim_3 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_3 (Create a shiny finish after use)', 'Claim_3 (Eco-friendly/Good for the environment)', 'Claim_3 (Fun designs/ shapes to elevate your mood)', 'Claim_3 (Guarantee satisfaction after use)', 'Claim_3 (Keeps your family safe)', 'Claim_3 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_3 (Scented (emits pleasant scent while cleaning))', 'Claim_3 (Transform cleaning into a fun experience with foamy bubble)', 'Material_3 (100% Recycled PET plastic)', 'Material_3 (Flexible foam (texture changes with water temperature))', 'Material_3 (Man-made fibre (e.g., polyester))', 'Material_3 (Melamine foam (magic eraser))', 'Material_3 (Natural cellulose sponge)', 'Material_3 (Natural coconut based fibre)', 'Material_3 (Natural corn based fibre)', 'Material_3 (Stay Fresh foam (foam that resist stains and odors))', 'Price_3', 'Color_3 (Beige/Brown )', 'Color_3 (Black )', 'Color_3 (Blue)', 'Color_3 (Dark Green)', 'Color_3 (Green)', 'Color_3 (Orange)', 'Color_3 (Pink)', 'Color_3 (Purple)', 'Color_3 (White)', 'Color_3 (Yellow)', 'Shape_3 (Flower)', 'Shape_3 (Leaf)', 'Shape_3 (Oval )', 'Shape_3 (Rectangle)', 'Shape_3 (Round)', 'Shape_3 (Smiley)', 'Shape_3 (Square)', 'Shape_3 (Tear drop )', 'Shape_3 (Wave )'],
	['Brand_4 (Byulpyo)', 'Brand_4 (Cleanwrap)', 'Brand_4 (Daiso)', 'Brand_4 (Frog)', 'Brand_4 (No Brand)', 'Brand_4 (Scotch-Brite by 3M)', 'Brand_4 (Scott (Yuhan-Kimberly))', 'Brand_4 (Spontex)', 'Cleaning strength_4 (Extra tough/ extra heavy duty)', 'Cleaning strength_4 (Gentle/ delicate cleaning)', 'Cleaning strength_4 (Tough Cleaning/ Heavy duty )', 'Surface types_4 (Cast Iron (Cookware/Grills/etc))', 'Surface types_4 (Everyday dishware & utensils)', 'Surface types_4 (Glass (Cooktop))', 'Surface types_4 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_4 (Plastic (Dishware, Utensils, etc))', 'Surface types_4 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_4 (Stainless Steel (Cooktop))', 'Surface types_4 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_4 (Stainless Steel (Sink /Cookware) )', 'Surface types_4 (Suitable for use across most surface types)', 'Stain types_4 (Effective at removing different stuck on and oily stains)', 'Stain types_4 (Good at removing heavy oil stains)', 'Stain types_4 (Good at removing light (non-oily) stains)', 'Stain types_4 (Good at removing light oil stains)', 'Stain types_4 (Good at removing sticky stains)', 'Stain types_4 (Good at removing stubborn burnt stains)', 'Stain types_4 (Good at removing tea/coffee stains)', 'Benefit_4 (Antibacterial/Prevent bacterial growth)', 'Benefit_4 (Antimicrobial/Prevent mold & odor)', 'Benefit_4 (Durable/long lasting)', 'Benefit_4 (Easy to grip)', 'Benefit_4 (Easy to reach and clean tight corners and grooves)', "Benefit_4 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_4 (Foam well with a small amount of detergent)', 'Benefit_4 (Stain resistant)', 'Benefit_4 (Will not scratch or damage cleaning surfaces)', 'Claim_4 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_4 (Create a shiny finish after use)', 'Claim_4 (Eco-friendly/Good for the environment)', 'Claim_4 (Fun designs/ shapes to elevate your mood)', 'Claim_4 (Guarantee satisfaction after use)', 'Claim_4 (Keeps your family safe)', 'Claim_4 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_4 (Scented (emits pleasant scent while cleaning))', 'Claim_4 (Transform cleaning into a fun experience with foamy bubble)', 'Material_4 (100% Recycled PET plastic)', 'Material_4 (Flexible foam (texture changes with water temperature))', 'Material_4 (Man-made fibre (e.g., polyester))', 'Material_4 (Melamine foam (magic eraser))', 'Material_4 (Natural cellulose sponge)', 'Material_4 (Natural coconut based fibre)', 'Material_4 (Natural corn based fibre)', 'Material_4 (Stay Fresh foam (foam that resist stains and odors))', 'Price_4', 'Color_4 (Beige/Brown )', 'Color_4 (Black )', 'Color_4 (Blue)', 'Color_4 (Dark Green)', 'Color_4 (Green)', 'Color_4 (Orange)', 'Color_4 (Pink)', 'Color_4 (Purple)', 'Color_4 (White)', 'Color_4 (Yellow)', 'Shape_4 (Flower)', 'Shape_4 (Leaf)', 'Shape_4 (Oval )', 'Shape_4 (Rectangle)', 'Shape_4 (Round)', 'Shape_4 (Smiley)', 'Shape_4 (Square)', 'Shape_4 (Tear drop )', 'Shape_4 (Wave )'],
	['Brand_5 (Byulpyo)', 'Brand_5 (Cleanwrap)', 'Brand_5 (Daiso)', 'Brand_5 (Frog)', 'Brand_5 (No Brand)', 'Brand_5 (Scotch-Brite by 3M)', 'Brand_5 (Scott (Yuhan-Kimberly))', 'Brand_5 (Spontex)', 'Cleaning strength_5 (Extra tough/ extra heavy duty)', 'Cleaning strength_5 (Gentle/ delicate cleaning)', 'Cleaning strength_5 (Tough Cleaning/ Heavy duty )', 'Surface types_5 (Cast Iron (Cookware/Grills/etc))', 'Surface types_5 (Everyday dishware & utensils)', 'Surface types_5 (Glass (Cooktop))', 'Surface types_5 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_5 (Plastic (Dishware, Utensils, etc))', 'Surface types_5 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_5 (Stainless Steel (Cooktop))', 'Surface types_5 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_5 (Stainless Steel (Sink /Cookware) )', 'Surface types_5 (Suitable for use across most surface types)', 'Stain types_5 (Effective at removing different stuck on and oily stains)', 'Stain types_5 (Good at removing heavy oil stains)', 'Stain types_5 (Good at removing light (non-oily) stains)', 'Stain types_5 (Good at removing light oil stains)', 'Stain types_5 (Good at removing sticky stains)', 'Stain types_5 (Good at removing stubborn burnt stains)', 'Stain types_5 (Good at removing tea/coffee stains)', 'Benefit_5 (Antibacterial/Prevent bacterial growth)', 'Benefit_5 (Antimicrobial/Prevent mold & odor)', 'Benefit_5 (Durable/long lasting)', 'Benefit_5 (Easy to grip)', 'Benefit_5 (Easy to reach and clean tight corners and grooves)', "Benefit_5 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_5 (Foam well with a small amount of detergent)', 'Benefit_5 (Stain resistant)', 'Benefit_5 (Will not scratch or damage cleaning surfaces)', 'Claim_5 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_5 (Create a shiny finish after use)', 'Claim_5 (Eco-friendly/Good for the environment)', 'Claim_5 (Fun designs/ shapes to elevate your mood)', 'Claim_5 (Guarantee satisfaction after use)', 'Claim_5 (Keeps your family safe)', 'Claim_5 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_5 (Scented (emits pleasant scent while cleaning))', 'Claim_5 (Transform cleaning into a fun experience with foamy bubble)', 'Material_5 (100% Recycled PET plastic)', 'Material_5 (Flexible foam (texture changes with water temperature))', 'Material_5 (Man-made fibre (e.g., polyester))', 'Material_5 (Melamine foam (magic eraser))', 'Material_5 (Natural cellulose sponge)', 'Material_5 (Natural coconut based fibre)', 'Material_5 (Natural corn based fibre)', 'Material_5 (Stay Fresh foam (foam that resist stains and odors))', 'Price_5', 'Color_5 (Beige/Brown )', 'Color_5 (Black )', 'Color_5 (Blue)', 'Color_5 (Dark Green)', 'Color_5 (Green)', 'Color_5 (Orange)', 'Color_5 (Pink)', 'Color_5 (Purple)', 'Color_5 (White)', 'Color_5 (Yellow)', 'Shape_5 (Flower)', 'Shape_5 (Leaf)', 'Shape_5 (Oval )', 'Shape_5 (Rectangle)', 'Shape_5 (Round)', 'Shape_5 (Smiley)', 'Shape_5 (Square)', 'Shape_5 (Tear drop )', 'Shape_5 (Wave )'],
	['Brand_6 (Byulpyo)', 'Brand_6 (Cleanwrap)', 'Brand_6 (Daiso)', 'Brand_6 (Frog)', 'Brand_6 (No Brand)', 'Brand_6 (Scotch-Brite by 3M)', 'Brand_6 (Scott (Yuhan-Kimberly))', 'Brand_6 (Spontex)', 'Cleaning strength_6 (Extra tough/ extra heavy duty)', 'Cleaning strength_6 (Gentle/ delicate cleaning)', 'Cleaning strength_6 (Tough Cleaning/ Heavy duty )', 'Surface types_6 (Cast Iron (Cookware/Grills/etc))', 'Surface types_6 (Everyday dishware & utensils)', 'Surface types_6 (Glass (Cooktop))', 'Surface types_6 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_6 (Plastic (Dishware, Utensils, etc))', 'Surface types_6 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_6 (Stainless Steel (Cooktop))', 'Surface types_6 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_6 (Stainless Steel (Sink /Cookware) )', 'Surface types_6 (Suitable for use across most surface types)', 'Stain types_6 (Effective at removing different stuck on and oily stains)', 'Stain types_6 (Good at removing heavy oil stains)', 'Stain types_6 (Good at removing light (non-oily) stains)', 'Stain types_6 (Good at removing light oil stains)', 'Stain types_6 (Good at removing sticky stains)', 'Stain types_6 (Good at removing stubborn burnt stains)', 'Stain types_6 (Good at removing tea/coffee stains)', 'Benefit_6 (Antibacterial/Prevent bacterial growth)', 'Benefit_6 (Antimicrobial/Prevent mold & odor)', 'Benefit_6 (Durable/long lasting)', 'Benefit_6 (Easy to grip)', 'Benefit_6 (Easy to reach and clean tight corners and grooves)', "Benefit_6 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_6 (Foam well with a small amount of detergent)', 'Benefit_6 (Stain resistant)', 'Benefit_6 (Will not scratch or damage cleaning surfaces)', 'Claim_6 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_6 (Create a shiny finish after use)', 'Claim_6 (Eco-friendly/Good for the environment)', 'Claim_6 (Fun designs/ shapes to elevate your mood)', 'Claim_6 (Guarantee satisfaction after use)', 'Claim_6 (Keeps your family safe)', 'Claim_6 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_6 (Scented (emits pleasant scent while cleaning))', 'Claim_6 (Transform cleaning into a fun experience with foamy bubble)', 'Material_6 (100% Recycled PET plastic)', 'Material_6 (Flexible foam (texture changes with water temperature))', 'Material_6 (Man-made fibre (e.g., polyester))', 'Material_6 (Melamine foam (magic eraser))', 'Material_6 (Natural cellulose sponge)', 'Material_6 (Natural coconut based fibre)', 'Material_6 (Natural corn based fibre)', 'Material_6 (Stay Fresh foam (foam that resist stains and odors))', 'Price_6', 'Color_6 (Beige/Brown )', 'Color_6 (Black )', 'Color_6 (Blue)', 'Color_6 (Dark Green)', 'Color_6 (Green)', 'Color_6 (Orange)', 'Color_6 (Pink)', 'Color_6 (Purple)', 'Color_6 (White)', 'Color_6 (Yellow)', 'Shape_6 (Flower)', 'Shape_6 (Leaf)', 'Shape_6 (Oval )', 'Shape_6 (Rectangle)', 'Shape_6 (Round)', 'Shape_6 (Smiley)', 'Shape_6 (Square)', 'Shape_6 (Tear drop )', 'Shape_6 (Wave )'],
	['Brand_7 (Byulpyo)', 'Brand_7 (Cleanwrap)', 'Brand_7 (Daiso)', 'Brand_7 (Frog)', 'Brand_7 (No Brand)', 'Brand_7 (Scotch-Brite by 3M)', 'Brand_7 (Scott (Yuhan-Kimberly))', 'Brand_7 (Spontex)', 'Cleaning strength_7 (Extra tough/ extra heavy duty)', 'Cleaning strength_7 (Gentle/ delicate cleaning)', 'Cleaning strength_7 (Tough Cleaning/ Heavy duty )', 'Surface types_7 (Cast Iron (Cookware/Grills/etc))', 'Surface types_7 (Everyday dishware & utensils)', 'Surface types_7 (Glass (Cooktop))', 'Surface types_7 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_7 (Plastic (Dishware, Utensils, etc))', 'Surface types_7 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_7 (Stainless Steel (Cooktop))', 'Surface types_7 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_7 (Stainless Steel (Sink /Cookware) )', 'Surface types_7 (Suitable for use across most surface types)', 'Stain types_7 (Effective at removing different stuck on and oily stains)', 'Stain types_7 (Good at removing heavy oil stains)', 'Stain types_7 (Good at removing light (non-oily) stains)', 'Stain types_7 (Good at removing light oil stains)', 'Stain types_7 (Good at removing sticky stains)', 'Stain types_7 (Good at removing stubborn burnt stains)', 'Stain types_7 (Good at removing tea/coffee stains)', 'Benefit_7 (Antibacterial/Prevent bacterial growth)', 'Benefit_7 (Antimicrobial/Prevent mold & odor)', 'Benefit_7 (Durable/long lasting)', 'Benefit_7 (Easy to grip)', 'Benefit_7 (Easy to reach and clean tight corners and grooves)', "Benefit_7 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_7 (Foam well with a small amount of detergent)', 'Benefit_7 (Stain resistant)', 'Benefit_7 (Will not scratch or damage cleaning surfaces)', 'Claim_7 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_7 (Create a shiny finish after use)', 'Claim_7 (Eco-friendly/Good for the environment)', 'Claim_7 (Fun designs/ shapes to elevate your mood)', 'Claim_7 (Guarantee satisfaction after use)', 'Claim_7 (Keeps your family safe)', 'Claim_7 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_7 (Scented (emits pleasant scent while cleaning))', 'Claim_7 (Transform cleaning into a fun experience with foamy bubble)', 'Material_7 (100% Recycled PET plastic)', 'Material_7 (Flexible foam (texture changes with water temperature))', 'Material_7 (Man-made fibre (e.g., polyester))', 'Material_7 (Melamine foam (magic eraser))', 'Material_7 (Natural cellulose sponge)', 'Material_7 (Natural coconut based fibre)', 'Material_7 (Natural corn based fibre)', 'Material_7 (Stay Fresh foam (foam that resist stains and odors))', 'Price_7', 'Color_7 (Beige/Brown )', 'Color_7 (Black )', 'Color_7 (Blue)', 'Color_7 (Dark Green)', 'Color_7 (Green)', 'Color_7 (Orange)', 'Color_7 (Pink)', 'Color_7 (Purple)', 'Color_7 (White)', 'Color_7 (Yellow)', 'Shape_7 (Flower)', 'Shape_7 (Leaf)', 'Shape_7 (Oval )', 'Shape_7 (Rectangle)', 'Shape_7 (Round)', 'Shape_7 (Smiley)', 'Shape_7 (Square)', 'Shape_7 (Tear drop )', 'Shape_7 (Wave )'],
	['Brand_8 (Byulpyo)', 'Brand_8 (Cleanwrap)', 'Brand_8 (Daiso)', 'Brand_8 (Frog)', 'Brand_8 (No Brand)', 'Brand_8 (Scotch-Brite by 3M)', 'Brand_8 (Scott (Yuhan-Kimberly))', 'Brand_8 (Spontex)', 'Cleaning strength_8 (Extra tough/ extra heavy duty)', 'Cleaning strength_8 (Gentle/ delicate cleaning)', 'Cleaning strength_8 (Tough Cleaning/ Heavy duty )', 'Surface types_8 (Cast Iron (Cookware/Grills/etc))', 'Surface types_8 (Everyday dishware & utensils)', 'Surface types_8 (Glass (Cooktop))', 'Surface types_8 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_8 (Plastic (Dishware, Utensils, etc))', 'Surface types_8 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_8 (Stainless Steel (Cooktop))', 'Surface types_8 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_8 (Stainless Steel (Sink /Cookware) )', 'Surface types_8 (Suitable for use across most surface types)', 'Stain types_8 (Effective at removing different stuck on and oily stains)', 'Stain types_8 (Good at removing heavy oil stains)', 'Stain types_8 (Good at removing light (non-oily) stains)', 'Stain types_8 (Good at removing light oil stains)', 'Stain types_8 (Good at removing sticky stains)', 'Stain types_8 (Good at removing stubborn burnt stains)', 'Stain types_8 (Good at removing tea/coffee stains)', 'Benefit_8 (Antibacterial/Prevent bacterial growth)', 'Benefit_8 (Antimicrobial/Prevent mold & odor)', 'Benefit_8 (Durable/long lasting)', 'Benefit_8 (Easy to grip)', 'Benefit_8 (Easy to reach and clean tight corners and grooves)', "Benefit_8 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_8 (Foam well with a small amount of detergent)', 'Benefit_8 (Stain resistant)', 'Benefit_8 (Will not scratch or damage cleaning surfaces)', 'Claim_8 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_8 (Create a shiny finish after use)', 'Claim_8 (Eco-friendly/Good for the environment)', 'Claim_8 (Fun designs/ shapes to elevate your mood)', 'Claim_8 (Guarantee satisfaction after use)', 'Claim_8 (Keeps your family safe)', 'Claim_8 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_8 (Scented (emits pleasant scent while cleaning))', 'Claim_8 (Transform cleaning into a fun experience with foamy bubble)', 'Material_8 (100% Recycled PET plastic)', 'Material_8 (Flexible foam (texture changes with water temperature))', 'Material_8 (Man-made fibre (e.g., polyester))', 'Material_8 (Melamine foam (magic eraser))', 'Material_8 (Natural cellulose sponge)', 'Material_8 (Natural coconut based fibre)', 'Material_8 (Natural corn based fibre)', 'Material_8 (Stay Fresh foam (foam that resist stains and odors))', 'Price_8', 'Color_8 (Beige/Brown )', 'Color_8 (Black )', 'Color_8 (Blue)', 'Color_8 (Dark Green)', 'Color_8 (Green)', 'Color_8 (Orange)', 'Color_8 (Pink)', 'Color_8 (Purple)', 'Color_8 (White)', 'Color_8 (Yellow)', 'Shape_8 (Flower)', 'Shape_8 (Leaf)', 'Shape_8 (Oval )', 'Shape_8 (Rectangle)', 'Shape_8 (Round)', 'Shape_8 (Smiley)', 'Shape_8 (Square)', 'Shape_8 (Tear drop )', 'Shape_8 (Wave )'],
	['Brand_9 (Byulpyo)', 'Brand_9 (Cleanwrap)', 'Brand_9 (Daiso)', 'Brand_9 (Frog)', 'Brand_9 (No Brand)', 'Brand_9 (Scotch-Brite by 3M)', 'Brand_9 (Scott (Yuhan-Kimberly))', 'Brand_9 (Spontex)', 'Cleaning strength_9 (Extra tough/ extra heavy duty)', 'Cleaning strength_9 (Gentle/ delicate cleaning)', 'Cleaning strength_9 (Tough Cleaning/ Heavy duty )', 'Surface types_9 (Cast Iron (Cookware/Grills/etc))', 'Surface types_9 (Everyday dishware & utensils)', 'Surface types_9 (Glass (Cooktop))', 'Surface types_9 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_9 (Plastic (Dishware, Utensils, etc))', 'Surface types_9 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_9 (Stainless Steel (Cooktop))', 'Surface types_9 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_9 (Stainless Steel (Sink /Cookware) )', 'Surface types_9 (Suitable for use across most surface types)', 'Stain types_9 (Effective at removing different stuck on and oily stains)', 'Stain types_9 (Good at removing heavy oil stains)', 'Stain types_9 (Good at removing light (non-oily) stains)', 'Stain types_9 (Good at removing light oil stains)', 'Stain types_9 (Good at removing sticky stains)', 'Stain types_9 (Good at removing stubborn burnt stains)', 'Stain types_9 (Good at removing tea/coffee stains)', 'Benefit_9 (Antibacterial/Prevent bacterial growth)', 'Benefit_9 (Antimicrobial/Prevent mold & odor)', 'Benefit_9 (Durable/long lasting)', 'Benefit_9 (Easy to grip)', 'Benefit_9 (Easy to reach and clean tight corners and grooves)', "Benefit_9 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_9 (Foam well with a small amount of detergent)', 'Benefit_9 (Stain resistant)', 'Benefit_9 (Will not scratch or damage cleaning surfaces)', 'Claim_9 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_9 (Create a shiny finish after use)', 'Claim_9 (Eco-friendly/Good for the environment)', 'Claim_9 (Fun designs/ shapes to elevate your mood)', 'Claim_9 (Guarantee satisfaction after use)', 'Claim_9 (Keeps your family safe)', 'Claim_9 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_9 (Scented (emits pleasant scent while cleaning))', 'Claim_9 (Transform cleaning into a fun experience with foamy bubble)', 'Material_9 (100% Recycled PET plastic)', 'Material_9 (Flexible foam (texture changes with water temperature))', 'Material_9 (Man-made fibre (e.g., polyester))', 'Material_9 (Melamine foam (magic eraser))', 'Material_9 (Natural cellulose sponge)', 'Material_9 (Natural coconut based fibre)', 'Material_9 (Natural corn based fibre)', 'Material_9 (Stay Fresh foam (foam that resist stains and odors))', 'Price_9', 'Color_9 (Beige/Brown )', 'Color_9 (Black )', 'Color_9 (Blue)', 'Color_9 (Dark Green)', 'Color_9 (Green)', 'Color_9 (Orange)', 'Color_9 (Pink)', 'Color_9 (Purple)', 'Color_9 (White)', 'Color_9 (Yellow)', 'Shape_9 (Flower)', 'Shape_9 (Leaf)', 'Shape_9 (Oval )', 'Shape_9 (Rectangle)', 'Shape_9 (Round)', 'Shape_9 (Smiley)', 'Shape_9 (Square)', 'Shape_9 (Tear drop )', 'Shape_9 (Wave )'],
	['Brand_10 (Byulpyo)', 'Brand_10 (Cleanwrap)', 'Brand_10 (Daiso)', 'Brand_10 (Frog)', 'Brand_10 (No Brand)', 'Brand_10 (Scotch-Brite by 3M)', 'Brand_10 (Scott (Yuhan-Kimberly))', 'Brand_10 (Spontex)', 'Cleaning strength_10 (Extra tough/ extra heavy duty)', 'Cleaning strength_10 (Gentle/ delicate cleaning)', 'Cleaning strength_10 (Tough Cleaning/ Heavy duty )', 'Surface types_10 (Cast Iron (Cookware/Grills/etc))', 'Surface types_10 (Everyday dishware & utensils)', 'Surface types_10 (Glass (Cooktop))', 'Surface types_10 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_10 (Plastic (Dishware, Utensils, etc))', 'Surface types_10 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_10 (Stainless Steel (Cooktop))', 'Surface types_10 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_10 (Stainless Steel (Sink /Cookware) )', 'Surface types_10 (Suitable for use across most surface types)', 'Stain types_10 (Effective at removing different stuck on and oily stains)', 'Stain types_10 (Good at removing heavy oil stains)', 'Stain types_10 (Good at removing light (non-oily) stains)', 'Stain types_10 (Good at removing light oil stains)', 'Stain types_10 (Good at removing sticky stains)', 'Stain types_10 (Good at removing stubborn burnt stains)', 'Stain types_10 (Good at removing tea/coffee stains)', 'Benefit_10 (Antibacterial/Prevent bacterial growth)', 'Benefit_10 (Antimicrobial/Prevent mold & odor)', 'Benefit_10 (Durable/long lasting)', 'Benefit_10 (Easy to grip)', 'Benefit_10 (Easy to reach and clean tight corners and grooves)', "Benefit_10 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_10 (Foam well with a small amount of detergent)', 'Benefit_10 (Stain resistant)', 'Benefit_10 (Will not scratch or damage cleaning surfaces)', 'Claim_10 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_10 (Create a shiny finish after use)', 'Claim_10 (Eco-friendly/Good for the environment)', 'Claim_10 (Fun designs/ shapes to elevate your mood)', 'Claim_10 (Guarantee satisfaction after use)', 'Claim_10 (Keeps your family safe)', 'Claim_10 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_10 (Scented (emits pleasant scent while cleaning))', 'Claim_10 (Transform cleaning into a fun experience with foamy bubble)', 'Material_10 (100% Recycled PET plastic)', 'Material_10 (Flexible foam (texture changes with water temperature))', 'Material_10 (Man-made fibre (e.g., polyester))', 'Material_10 (Melamine foam (magic eraser))', 'Material_10 (Natural cellulose sponge)', 'Material_10 (Natural coconut based fibre)', 'Material_10 (Natural corn based fibre)', 'Material_10 (Stay Fresh foam (foam that resist stains and odors))', 'Price_10', 'Color_10 (Beige/Brown )', 'Color_10 (Black )', 'Color_10 (Blue)', 'Color_10 (Dark Green)', 'Color_10 (Green)', 'Color_10 (Orange)', 'Color_10 (Pink)', 'Color_10 (Purple)', 'Color_10 (White)', 'Color_10 (Yellow)', 'Shape_10 (Flower)', 'Shape_10 (Leaf)', 'Shape_10 (Oval )', 'Shape_10 (Rectangle)', 'Shape_10 (Round)', 'Shape_10 (Smiley)', 'Shape_10 (Square)', 'Shape_10 (Tear drop )', 'Shape_10 (Wave )'],
	['Brand_11 (Byulpyo)', 'Brand_11 (Cleanwrap)', 'Brand_11 (Daiso)', 'Brand_11 (Frog)', 'Brand_11 (No Brand)', 'Brand_11 (Scotch-Brite by 3M)', 'Brand_11 (Scott (Yuhan-Kimberly))', 'Brand_11 (Spontex)', 'Cleaning strength_11 (Extra tough/ extra heavy duty)', 'Cleaning strength_11 (Gentle/ delicate cleaning)', 'Cleaning strength_11 (Tough Cleaning/ Heavy duty )', 'Surface types_11 (Cast Iron (Cookware/Grills/etc))', 'Surface types_11 (Everyday dishware & utensils)', 'Surface types_11 (Glass (Cooktop))', 'Surface types_11 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_11 (Plastic (Dishware, Utensils, etc))', 'Surface types_11 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_11 (Stainless Steel (Cooktop))', 'Surface types_11 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_11 (Stainless Steel (Sink /Cookware) )', 'Surface types_11 (Suitable for use across most surface types)', 'Stain types_11 (Effective at removing different stuck on and oily stains)', 'Stain types_11 (Good at removing heavy oil stains)', 'Stain types_11 (Good at removing light (non-oily) stains)', 'Stain types_11 (Good at removing light oil stains)', 'Stain types_11 (Good at removing sticky stains)', 'Stain types_11 (Good at removing stubborn burnt stains)', 'Stain types_11 (Good at removing tea/coffee stains)', 'Benefit_11 (Antibacterial/Prevent bacterial growth)', 'Benefit_11 (Antimicrobial/Prevent mold & odor)', 'Benefit_11 (Durable/long lasting)', 'Benefit_11 (Easy to grip)', 'Benefit_11 (Easy to reach and clean tight corners and grooves)', "Benefit_11 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_11 (Foam well with a small amount of detergent)', 'Benefit_11 (Stain resistant)', 'Benefit_11 (Will not scratch or damage cleaning surfaces)', 'Claim_11 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_11 (Create a shiny finish after use)', 'Claim_11 (Eco-friendly/Good for the environment)', 'Claim_11 (Fun designs/ shapes to elevate your mood)', 'Claim_11 (Guarantee satisfaction after use)', 'Claim_11 (Keeps your family safe)', 'Claim_11 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_11 (Scented (emits pleasant scent while cleaning))', 'Claim_11 (Transform cleaning into a fun experience with foamy bubble)', 'Material_11 (100% Recycled PET plastic)', 'Material_11 (Flexible foam (texture changes with water temperature))', 'Material_11 (Man-made fibre (e.g., polyester))', 'Material_11 (Melamine foam (magic eraser))', 'Material_11 (Natural cellulose sponge)', 'Material_11 (Natural coconut based fibre)', 'Material_11 (Natural corn based fibre)', 'Material_11 (Stay Fresh foam (foam that resist stains and odors))', 'Price_11', 'Color_11 (Beige/Brown )', 'Color_11 (Black )', 'Color_11 (Blue)', 'Color_11 (Dark Green)', 'Color_11 (Green)', 'Color_11 (Orange)', 'Color_11 (Pink)', 'Color_11 (Purple)', 'Color_11 (White)', 'Color_11 (Yellow)', 'Shape_11 (Flower)', 'Shape_11 (Leaf)', 'Shape_11 (Oval )', 'Shape_11 (Rectangle)', 'Shape_11 (Round)', 'Shape_11 (Smiley)', 'Shape_11 (Square)', 'Shape_11 (Tear drop )', 'Shape_11 (Wave )'],
	['Brand_12 (Byulpyo)', 'Brand_12 (Cleanwrap)', 'Brand_12 (Daiso)', 'Brand_12 (Frog)', 'Brand_12 (No Brand)', 'Brand_12 (Scotch-Brite by 3M)', 'Brand_12 (Scott (Yuhan-Kimberly))', 'Brand_12 (Spontex)', 'Cleaning strength_12 (Extra tough/ extra heavy duty)', 'Cleaning strength_12 (Gentle/ delicate cleaning)', 'Cleaning strength_12 (Tough Cleaning/ Heavy duty )', 'Surface types_12 (Cast Iron (Cookware/Grills/etc))', 'Surface types_12 (Everyday dishware & utensils)', 'Surface types_12 (Glass (Cooktop))', 'Surface types_12 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_12 (Plastic (Dishware, Utensils, etc))', 'Surface types_12 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_12 (Stainless Steel (Cooktop))', 'Surface types_12 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_12 (Stainless Steel (Sink /Cookware) )', 'Surface types_12 (Suitable for use across most surface types)', 'Stain types_12 (Effective at removing different stuck on and oily stains)', 'Stain types_12 (Good at removing heavy oil stains)', 'Stain types_12 (Good at removing light (non-oily) stains)', 'Stain types_12 (Good at removing light oil stains)', 'Stain types_12 (Good at removing sticky stains)', 'Stain types_12 (Good at removing stubborn burnt stains)', 'Stain types_12 (Good at removing tea/coffee stains)', 'Benefit_12 (Antibacterial/Prevent bacterial growth)', 'Benefit_12 (Antimicrobial/Prevent mold & odor)', 'Benefit_12 (Durable/long lasting)', 'Benefit_12 (Easy to grip)', 'Benefit_12 (Easy to reach and clean tight corners and grooves)', "Benefit_12 (Easy to rinse clean after use (food doesn't get stuck))", 'Benefit_12 (Foam well with a small amount of detergent)', 'Benefit_12 (Stain resistant)', 'Benefit_12 (Will not scratch or damage cleaning surfaces)', 'Claim_12 (Convenient as no soap required (infused with  cleaning detergent))', 'Claim_12 (Create a shiny finish after use)', 'Claim_12 (Eco-friendly/Good for the environment)', 'Claim_12 (Fun designs/ shapes to elevate your mood)', 'Claim_12 (Guarantee satisfaction after use)', 'Claim_12 (Keeps your family safe)', 'Claim_12 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_12 (Scented (emits pleasant scent while cleaning))', 'Claim_12 (Transform cleaning into a fun experience with foamy bubble)', 'Material_12 (100% Recycled PET plastic)', 'Material_12 (Flexible foam (texture changes with water temperature))', 'Material_12 (Man-made fibre (e.g., polyester))', 'Material_12 (Melamine foam (magic eraser))', 'Material_12 (Natural cellulose sponge)', 'Material_12 (Natural coconut based fibre)', 'Material_12 (Natural corn based fibre)', 'Material_12 (Stay Fresh foam (foam that resist stains and odors))', 'Price_12', 'Color_12 (Beige/Brown )', 'Color_12 (Black )', 'Color_12 (Blue)', 'Color_12 (Dark Green)', 'Color_12 (Green)', 'Color_12 (Orange)', 'Color_12 (Pink)', 'Color_12 (Purple)', 'Color_12 (White)', 'Color_12 (Yellow)', 'Shape_12 (Flower)', 'Shape_12 (Leaf)', 'Shape_12 (Oval )', 'Shape_12 (Rectangle)', 'Shape_12 (Round)', 'Shape_12 (Smiley)', 'Shape_12 (Square)', 'Shape_12 (Tear drop )', 'Shape_12 (Wave )'],
	]

		Ut_Demo_Variables = ['Gender (Male)', 'Income (6,000,000 KRW – 6,999,999 KRW)',
						 'Income (4,000,000 KRW – 4,999,999 KRW)', 'Income (7,000,000 KRW – 7,999,999 KRW)',
						 'Income (8,000,000 KRW – 8,999,999 KRW)', 'Income (5,000,000 KRW – 5,999,999 KRW)',
						 'Income (2,000,000 KRW – 2,999,999 KRW)', 'Income (9,000,000 KRW – 9,999,999 KRW)',
						 'Income (10,000,000 KRW and above)', 'Decision_Maker (I sometimes decide)',
						 'Brand often brought (3M)', 'Brand often brought (Scotch-Brite by 3M)',
						 'Brand often brought (Cleanwrap)', 'Brand often brought (Daiso)',
						 'Brand often brought (Kitchen-Art)', 'Brand often brought (Scrub daddy)',
						 'Brand often brought (No Brand)', 'Brand often brought (Komax)', 'Brand often brought (Atomi)',
						 'Brand often brought (Trista (formerly Byulpyo))', 'Brand often brought (Scott (Yuhan-Kimberly))',
						 'Brand often brought (HomePlus PB)', 'Brand often brought (Amway)',
						 'Brand often brought (Spontex)', 'Brand often brought (HankookTamina)',
						 'Brand often brought (Loving Home(Emart PB))', 'Brand often brought (Frog)',
						 'Brand often brought (Svinto)', 'Brand often brought (Lotte e-life)',
						 'Brand often brought (LotteAluminum)', 'Brand often brought (Wisewipe)',
						 'Brand often brought (Dearcus)', 'Brand often brought (Skrubba)']
	else:
		All_UT_Spec = [
			['Brand_1 (Scotch-Brite by 3M)', 'Brand_1 (Byulpyo)', 'Brand_1 (Cleanwrap)', 'Brand_1 (Frog)',
					 'Brand_1 (No Brand)', 'Brand_1 (Scott (Yuhan-Kimberly))', 'Brand_1 (Spontex)', 'Brand_1 (Daiso)',
					 'Format_1 (Sponge)', 'Format_1 (Scrub Sponge)', 'Format_1 (3 layers/2-sided scrub sponge)',
					 'Format_1 (Scrub Pad)', 'Format_1 (Large sheet/wipe)', 'Format_1 (Net/Mesh cloth )',
					 'Format_1 (Bottle cleaner with handle)', 'Format_1 (Acrylic scourer)', 'Format_1 (Handled Dishwand)',
					 'Format_1 (Disposable Sheet (Good for one/several washes))', 'Format_1 (Metal Ball Scourer)',
					 'Cleaning strength_1 (Extra tough/ extra heavy duty)',
					 'Cleaning strength_1 (Tough Cleaning/ Heavy duty )', 'Cleaning strength_1 (Gentle/ delicate cleaning)',
					 'Surface types_1 (Stainless Steel (Sink /Cookware) )',
					 'Surface types_1 (Stainless Steel (Outdoor camping/BBQ))',
					 'Surface types_1 (Stainless Steel (Cooktop))',
					 'Surface types_1 (Non-Stick/Ceramic (Cookware, Dishware, etc))',
					 'Surface types_1 (Cast Iron (Cookware/Grills/etc))',
					 'Surface types_1 (Porcelain (Dishware, premium crockery, sink, etc))',
					 'Surface types_1 (Glass (Cooktop))', 'Surface types_1 (Plastic (Dishware, Utensils, etc))',
					 'Surface types_1 (Everyday dishware & utensils)',
					 'Surface types_1 (Suitable for use across most surface types)',
					 'Stain types_1 (Good at removing light (non-oily) stains)',
					 'Stain types_1 (Good at removing tea/coffee stains)',
					 'Stain types_1 (Good at removing light oil stains)',
					 'Stain types_1 (Good at removing heavy oil stains)', 'Stain types_1 (Good at removing sticky stains)',
					 'Stain types_1 (Good at removing stubborn burnt stains)',
					 'Stain types_1 (Effective at removing different stuck on and oily stains)',
					 'Benefit_1 (Will not scratch or damage cleaning surfaces)', 'Benefit_1 (Durable/long lasting)',
					 'Benefit_1 (Antibacterial/Prevent bacterial growth)',
					 'Benefit_1 (Foam well with a small amount of detergent)', 'Benefit_1 (Easy to grip)',
					 "Benefit_1 (Easy to rinse clean after use (food doesn't get stuck))",
					 'Benefit_1 (Easy to reach and clean tight corners and grooves)',
					 'Benefit_1 (Antimicrobial/Prevent mold & odor)', 'Benefit_1 (Stain resistant)',
					 'Claim_1 (Make cleaning a breeze, saves time to do the things you love )',
					 'Claim_1 (Transform cleaning into a fun experience with foamy bubble)',
					 'Claim_1 (Create a shiny finish after use)', 'Claim_1 (Guarantee satisfaction after use)',
					 'Claim_1 (Eco-friendly/Good for the environment)', 'Claim_1 (Keeps your family safe)',
					 'Claim_1 (Fun designs/ shapes to elevate your mood)',
					 'Claim_1 (Scented (emits pleasant scent while cleaning))',
					 'Claim_1 (Convenient as no soap required (infused with  cleaning detergent))',
					 'Material_1 (Man-made fibre (e.g., polyester))', 'Material_1 (100% Recycled PET plastic)',
					 'Material_1 (Melamine foam (magic eraser))', 'Material_1 (Natural corn based fibre)',
					 'Material_1 (Natural coconut based fibre)', 'Material_1 (Natural cellulose sponge)',
					 'Material_1 (Flexible foam (texture changes with water temperature))',
					 'Material_1 (Stay Fresh foam (foam that resist stains and odors))', 'Price_1', 'Color_1 (Green)',
					 'Color_1 (Dark Green)', 'Color_1 (Yellow)', 'Color_1 (Orange)', 'Color_1 (Pink)', 'Color_1 (Blue)',
					 'Color_1 (Purple)', 'Color_1 (Beige/Brown )', 'Color_1 (Black )', 'Color_1 (White)',
					 'Shape_1 (Rectangle)', 'Shape_1 (Wave )', 'Shape_1 (Tear drop )', 'Shape_1 (Round)', 'Shape_1 (Leaf)',
					 'Shape_1 (Square)', 'Shape_1 (Oval )', 'Shape_1 (Smiley)', 'Shape_1 (Flower)'],
		['Brand_2 (Scotch-Brite by 3M)', 'Brand_2 (Byulpyo)', 'Brand_2 (Cleanwrap)', 'Brand_2 (Frog)',
				 'Brand_2 (No Brand)', 'Brand_2 (Scott (Yuhan-Kimberly))', 'Brand_2 (Spontex)', 'Brand_2 (Daiso)',
				 'Format_2 (Sponge)', 'Format_2 (Scrub Sponge)', 'Format_2 (3 layers/2-sided scrub sponge)',
				 'Format_2 (Scrub Pad)', 'Format_2 (Large sheet/wipe)', 'Format_2 (Net/Mesh cloth )',
				 'Format_2 (Bottle cleaner with handle)', 'Format_2 (Acrylic scourer)', 'Format_2 (Handled Dishwand)',
				 'Format_2 (Disposable Sheet (Good for one/several washes))', 'Format_2 (Metal Ball Scourer)',
				 'Cleaning strength_2 (Extra tough/ extra heavy duty)', 'Cleaning strength_2 (Tough Cleaning/ Heavy duty )',
				 'Cleaning strength_2 (Gentle/ delicate cleaning)', 'Surface types_2 (Stainless Steel (Sink /Cookware) )',
				 'Surface types_2 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_2 (Stainless Steel (Cooktop))',
				 'Surface types_2 (Non-Stick/Ceramic (Cookware, Dishware, etc))',
				 'Surface types_2 (Cast Iron (Cookware/Grills/etc))',
				 'Surface types_2 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_2 (Glass (Cooktop))',
				 'Surface types_2 (Plastic (Dishware, Utensils, etc))', 'Surface types_2 (Everyday dishware & utensils)',
				 'Surface types_2 (Suitable for use across most surface types)',
				 'Stain types_2 (Good at removing light (non-oily) stains)',
				 'Stain types_2 (Good at removing tea/coffee stains)', 'Stain types_2 (Good at removing light oil stains)',
				 'Stain types_2 (Good at removing heavy oil stains)', 'Stain types_2 (Good at removing sticky stains)',
				 'Stain types_2 (Good at removing stubborn burnt stains)',
				 'Stain types_2 (Effective at removing different stuck on and oily stains)',
				 'Benefit_2 (Will not scratch or damage cleaning surfaces)', 'Benefit_2 (Durable/long lasting)',
				 'Benefit_2 (Antibacterial/Prevent bacterial growth)',
				 'Benefit_2 (Foam well with a small amount of detergent)', 'Benefit_2 (Easy to grip)',
				 "Benefit_2 (Easy to rinse clean after use (food doesn't get stuck))",
				 'Benefit_2 (Easy to reach and clean tight corners and grooves)',
				 'Benefit_2 (Antimicrobial/Prevent mold & odor)', 'Benefit_2 (Stain resistant)',
				 'Claim_2 (Make cleaning a breeze, saves time to do the things you love )',
				 'Claim_2 (Transform cleaning into a fun experience with foamy bubble)',
				 'Claim_2 (Create a shiny finish after use)', 'Claim_2 (Guarantee satisfaction after use)',
				 'Claim_2 (Eco-friendly/Good for the environment)', 'Claim_2 (Keeps your family safe)',
				 'Claim_2 (Fun designs/ shapes to elevate your mood)',
				 'Claim_2 (Scented (emits pleasant scent while cleaning))',
				 'Claim_2 (Convenient as no soap required (infused with  cleaning detergent))',
				 'Material_2 (Man-made fibre (e.g., polyester))', 'Material_2 (100% Recycled PET plastic)',
				 'Material_2 (Melamine foam (magic eraser))', 'Material_2 (Natural corn based fibre)',
				 'Material_2 (Natural coconut based fibre)', 'Material_2 (Natural cellulose sponge)',
				 'Material_2 (Flexible foam (texture changes with water temperature))',
				 'Material_2 (Stay Fresh foam (foam that resist stains and odors))', 'Price_2', 'Color_2 (Green)',
				 'Color_2 (Dark Green)', 'Color_2 (Yellow)', 'Color_2 (Orange)', 'Color_2 (Pink)', 'Color_2 (Blue)',
				 'Color_2 (Purple)', 'Color_2 (Beige/Brown )', 'Color_2 (Black )', 'Color_2 (White)', 'Shape_2 (Rectangle)',
				 'Shape_2 (Wave )', 'Shape_2 (Tear drop )', 'Shape_2 (Round)', 'Shape_2 (Leaf)', 'Shape_2 (Square)',
				 'Shape_2 (Oval )', 'Shape_2 (Smiley)', 'Shape_2 (Flower)'],
		['Brand_3 (Scotch-Brite by 3M)', 'Brand_3 (Byulpyo)', 'Brand_3 (Cleanwrap)', 'Brand_3 (Frog)',
				 'Brand_3 (No Brand)', 'Brand_3 (Scott (Yuhan-Kimberly))', 'Brand_3 (Spontex)', 'Brand_3 (Daiso)',
				 'Format_3 (Sponge)', 'Format_3 (Scrub Sponge)', 'Format_3 (3 layers/2-sided scrub sponge)',
				 'Format_3 (Scrub Pad)', 'Format_3 (Large sheet/wipe)', 'Format_3 (Net/Mesh cloth )',
				 'Format_3 (Bottle cleaner with handle)', 'Format_3 (Acrylic scourer)', 'Format_3 (Handled Dishwand)',
				 'Format_3 (Disposable Sheet (Good for one/several washes))', 'Format_3 (Metal Ball Scourer)',
				 'Cleaning strength_3 (Extra tough/ extra heavy duty)', 'Cleaning strength_3 (Tough Cleaning/ Heavy duty )',
				 'Cleaning strength_3 (Gentle/ delicate cleaning)', 'Surface types_3 (Stainless Steel (Sink /Cookware) )',
				 'Surface types_3 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_3 (Stainless Steel (Cooktop))',
				 'Surface types_3 (Non-Stick/Ceramic (Cookware, Dishware, etc))',
				 'Surface types_3 (Cast Iron (Cookware/Grills/etc))',
				 'Surface types_3 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_3 (Glass (Cooktop))',
				 'Surface types_3 (Plastic (Dishware, Utensils, etc))', 'Surface types_3 (Everyday dishware & utensils)',
				 'Surface types_3 (Suitable for use across most surface types)',
				 'Stain types_3 (Good at removing light (non-oily) stains)',
				 'Stain types_3 (Good at removing tea/coffee stains)', 'Stain types_3 (Good at removing light oil stains)',
				 'Stain types_3 (Good at removing heavy oil stains)', 'Stain types_3 (Good at removing sticky stains)',
				 'Stain types_3 (Good at removing stubborn burnt stains)',
				 'Stain types_3 (Effective at removing different stuck on and oily stains)',
				 'Benefit_3 (Will not scratch or damage cleaning surfaces)', 'Benefit_3 (Durable/long lasting)',
				 'Benefit_3 (Antibacterial/Prevent bacterial growth)',
				 'Benefit_3 (Foam well with a small amount of detergent)', 'Benefit_3 (Easy to grip)',
				 "Benefit_3 (Easy to rinse clean after use (food doesn't get stuck))",
				 'Benefit_3 (Easy to reach and clean tight corners and grooves)',
				 'Benefit_3 (Antimicrobial/Prevent mold & odor)', 'Benefit_3 (Stain resistant)',
				 'Claim_3 (Make cleaning a breeze, saves time to do the things you love )',
				 'Claim_3 (Transform cleaning into a fun experience with foamy bubble)',
				 'Claim_3 (Create a shiny finish after use)', 'Claim_3 (Guarantee satisfaction after use)',
				 'Claim_3 (Eco-friendly/Good for the environment)', 'Claim_3 (Keeps your family safe)',
				 'Claim_3 (Fun designs/ shapes to elevate your mood)',
				 'Claim_3 (Scented (emits pleasant scent while cleaning))',
				 'Claim_3 (Convenient as no soap required (infused with  cleaning detergent))',
				 'Material_3 (Man-made fibre (e.g., polyester))', 'Material_3 (100% Recycled PET plastic)',
				 'Material_3 (Melamine foam (magic eraser))', 'Material_3 (Natural corn based fibre)',
				 'Material_3 (Natural coconut based fibre)', 'Material_3 (Natural cellulose sponge)',
				 'Material_3 (Flexible foam (texture changes with water temperature))',
				 'Material_3 (Stay Fresh foam (foam that resist stains and odors))', 'Price_3', 'Color_3 (Green)',
				 'Color_3 (Dark Green)', 'Color_3 (Yellow)', 'Color_3 (Orange)', 'Color_3 (Pink)', 'Color_3 (Blue)',
				 'Color_3 (Purple)', 'Color_3 (Beige/Brown )', 'Color_3 (Black )', 'Color_3 (White)', 'Shape_3 (Rectangle)',
				 'Shape_3 (Wave )', 'Shape_3 (Tear drop )', 'Shape_3 (Round)', 'Shape_3 (Leaf)', 'Shape_3 (Square)',
				 'Shape_3 (Oval )', 'Shape_3 (Smiley)', 'Shape_3 (Flower)'],

		]

	Alternative_Specific_Config          = True
	Add_Demo_Variables                   = True
	Add_Demo_Alternative_Specific_Config = True
	Add_Constant                         = True

	message_formatted5 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Generating Utility Specifications</p>'
	dp_progress.markdown(message_formatted5, unsafe_allow_html=True)

	if Model_Config != 'Labeled':
		Alternative_Specific_Config          = False
		Add_Demo_Variables                   = False
		Add_Demo_Alternative_Specific_Config = False
		Add_Constant                         = False

	Beta_Active = []
	if Alternative_Specific_Config:
		All_UT_Spec_New = []
		for index_lst, lst in enumerate(All_UT_Spec):
			temp = ['sero'] * len(lst) * nc
			st_index = index_lst * len(lst)
			for i in range(len(lst)):
				temp[st_index + i] = lst[i]
				check = Main_data[lst[i]].sum()
				if check >= 25:
					Beta_Active.append(1)
				else:
					Beta_Active.append(0)
			All_UT_Spec_New.append(temp)
		All_UT_Spec = All_UT_Spec_New
	else:
		Beta_Active = [1] * len(All_UT_Spec[0])

	if Add_Demo_Variables:
		All_UT_Spec_New = []
		if Add_Demo_Alternative_Specific_Config:
			for index_lst, lst in enumerate(All_UT_Spec):
				if index_lst == 0:
					temp = All_UT_Spec[index_lst] + ['sero'] * len(Ut_Demo_Variables) * (nc-1)
				else:
					temp = ['sero'] * len(Ut_Demo_Variables) * (nc-1)
					st_index = (index_lst-1) * len(Ut_Demo_Variables)
					for i in range(len(Ut_Demo_Variables)):
						temp[st_index + i] = Ut_Demo_Variables[i]
						check = Main_data[Ut_Demo_Variables[i]].sum()
						if check >= 25:
							Beta_Active.append(1)
						else:
							Beta_Active.append(0)
					temp = All_UT_Spec[index_lst] + temp
				All_UT_Spec_New.append(temp)
		else:
			for i in range(len(All_UT_Spec)):
				if i == 0:
					temp = All_UT_Spec[i] + ['sero'] * len(Ut_Demo_Variables)
				else:
					temp = All_UT_Spec[i] + Ut_Demo_Variables
				All_UT_Spec_New.append(temp)
			Beta_Active = Beta_Active + [1] * len(Ut_Demo_Variables)
		All_UT_Spec = All_UT_Spec_New

	if Add_Constant:
		All_UT_Spec_New = []
		for i in range(len(All_UT_Spec)):
			if i == 0:
				temp = ['sero'] * (nc - 1) + All_UT_Spec[i]
			else:
				temp = ['sero'] * (nc - 1)
				temp[i - 1] = 'uno'
				temp += All_UT_Spec[i]
			All_UT_Spec_New.append(temp)
		All_UT_Spec = All_UT_Spec_New
		Beta_Active = [1]*(nc - 1) + Beta_Active

	global nvarma_rum, ivgenva_chq, nvarma_chq, var_ind_chq, var_cutoff_type, var_cutoff_cumsum, var_parm_ind_chq, nvarma_chq_ind, nvarma_chq_ind_cumsum
	global Joint_Ch_Parm_Matrix, nvarma_chq_max, nvarma_chq_total_cumsum, nvarma_precision, nvarma_chq_actual, nvarma_chq_actual_csum, nvarma_rum_total
	ivgenva_rum = [item for sublist in All_UT_Spec for item in sublist]
	nvarma = len(All_UT_Spec[0])  # Getting the number of variables to be estimated for each dependent variable in the weighted utility specification component


	if Model_Config != 'Labeled':
		Output_Full_Name = ['Brand_1 (Scotch-Brite by 3M)', 'Brand_1 (Byulpyo)', 'Brand_1 (Cleanwrap)', 'Brand_1 (Frog)', 'Brand_1 (No Brand)', 'Brand_1 (Scott (Yuhan-Kimberly))', 'Brand_1 (Spontex)', 'Brand_1 (Daiso)', 'Brand_1 (Scrub Daddy)', 'Format_1 (Sponge)', 'Format_1 (Net Sponge)', 'Format_1 (Scrub Sponge)', 'Format_1 (3 layers/2-sided scrub sponge)', 'Format_1 (Scrub Pad)', 'Format_1 (Large sheet/wipe)', 'Format_1 (Net/Mesh cloth )', 'Format_1 (Bottle cleaner with handle)', 'Format_1 (Acrylic scourer)', 'Format_1 (Handled Dishwand)', 'Format_1 (Disposable Sheet (Good for one/several washes))', 'Format_1 (Metal Ball Scourer)', 'Cleaning strength_1 (Extra tough/ extra heavy duty)', 'Cleaning strength_1 (Tough Cleaning/ Heavy duty )', 'Cleaning strength_1 (Everyday Cleaning)', 'Cleaning strength_1 (Gentle/ delicate cleaning)', 'Surface types_1 (Stainless Steel (Sink /Cookware) )', 'Surface types_1 (Stainless Steel (Outdoor camping/BBQ))', 'Surface types_1 (Stainless Steel (Cooktop))', 'Surface types_1 (Non-Stick/Ceramic (Cookware, Dishware, etc))', 'Surface types_1 (Cast Iron (Cookware/Grills/etc))', 'Surface types_1 (Porcelain (Dishware, premium crockery, sink, etc))', 'Surface types_1 (Glass (Cookware/ Dishware))', 'Surface types_1 (Glass (Cooktop))', 'Surface types_1 (Plastic (Dishware, Utensils, etc))', 'Surface types_1 (Everyday dishware & utensils)', 'Surface types_1 (Suitable for use across most surface types)', 'Stain types_1 (Good at removing watermarks)', 'Stain types_1 (Good at removing light (non-oily) stains)', 'Stain types_1 (Good at removing tea/coffee stains)', 'Stain types_1 (Good at removing light oil stains)', 'Stain types_1 (Good at removing heavy oil stains)', 'Stain types_1 (Good at removing sticky stains)', 'Stain types_1 (Good at removing stubborn burnt stains)', 'Stain types_1 (Effective at removing different stuck on and oily stains)', 'Benefit_1 (Will not scratch or damage cleaning surfaces)', 'Benefit_1 (Durable/long lasting)', 'Benefit_1 (Antibacterial/Prevent bacterial growth)', 'Benefit_1 (Foam well with a small amount of detergent)', 'Benefit_1 (Easy to grip)', "Benefit_1 (Easy to rinse clean after use (food doesn't get stuck))", "Benefit_1 (Quick to dry after use (doesn't retain water))", 'Benefit_1 (Easy to reach and clean tight corners and grooves)', 'Benefit_1 (Antimicrobial/Prevent mold & odor)', 'Benefit_1 (Stain resistant)', 'Claim_1 (Make cleaning a breeze, saves time to do the things you love )', 'Claim_1 (Transform cleaning into a fun experience with foamy bubble)', 'Claim_1 (Create a shiny finish after use)', 'Claim_1 (Guarantee satisfaction after use)', 'Claim_1 (Comfortable to use )', 'Claim_1 (Eco-friendly/Good for the environment)', 'Claim_1 (Keeps your family safe)', 'Claim_1 (Fun designs/ shapes to elevate your mood)', 'Claim_1 (Scented (emits pleasant scent while cleaning))', 'Claim_1 (Convenient as no soap required (infused with  cleaning detergent))', 'Material_1 (Metal (e.g. Stainless Steel))', 'Material_1 (Man-made fibre (e.g., polyester))', 'Material_1 (100% Recycled PET plastic)', 'Material_1 (Melamine foam (magic eraser))', 'Material_1 (Natural corn based fibre)', 'Material_1 (Natural coconut based fibre)', 'Material_1 (Natural cellulose sponge)', 'Material_1 (Flexible foam (texture changes with water temperature))', 'Material_1 (Stay Fresh foam (foam that resist stains and odors))', 'Price_1', 'Color_1 (Green)', 'Color_1 (Dark Green)', 'Color_1 (Yellow)', 'Color_1 (Orange)', 'Color_1 (Pink)', 'Color_1 (Blue)', 'Color_1 (Purple)', 'Color_1 (Beige/Brown )', 'Color_1 (Black )', 'Color_1 (White)', 'Color_1 (Grey)', 'Shape_1 (Rectangle)', 'Shape_1 (Wave )', 'Shape_1 (Tear drop )', 'Shape_1 (Round)', 'Shape_1 (Leaf)', 'Shape_1 (Fish -shape)', 'Shape_1 (Square)', 'Shape_1 (Oval )', 'Shape_1 (Smiley)', 'Shape_1 (Flower)']
		Full_name_list = []
		for ele in Output_Full_Name:
			temp1, temp2 = split_brand_string(ele)
			Full_name_list.append([temp1, temp2])
	else:
		Output_Full_Name = ['Brand_1 (Scotch-Brite by 3M)', 'Brand_1 (Byulpyo)', 'Brand_1 (Cleanwrap)', 'Brand_1 (Frog)',
							'Brand_1 (No Brand)', 'Brand_1 (Scott (Yuhan-Kimberly))', 'Brand_1 (Spontex)',
							'Brand_1 (Daiso)', 'Brand_1 (Scrub Daddy)', 'Format_1 (Sponge)', 'Format_1 (Net Sponge)',
							'Format_1 (Scrub Sponge)', 'Format_1 (3 layers/2-sided scrub sponge)', 'Format_1 (Scrub Pad)',
							'Format_1 (Large sheet/wipe)', 'Format_1 (Net/Mesh cloth )',
							'Format_1 (Bottle cleaner with handle)', 'Format_1 (Acrylic scourer)',
							'Format_1 (Handled Dishwand)', 'Format_1 (Disposable Sheet (Good for one/several washes))',
							'Format_1 (Metal Ball Scourer)', 'Cleaning strength_1 (Extra tough/ extra heavy duty)',
							'Cleaning strength_1 (Tough Cleaning/ Heavy duty )', 'Cleaning strength_1 (Everyday Cleaning)',
							'Cleaning strength_1 (Gentle/ delicate cleaning)',
							'Surface types_1 (Stainless Steel (Sink /Cookware) )',
							'Surface types_1 (Stainless Steel (Outdoor camping/BBQ))',
							'Surface types_1 (Stainless Steel (Cooktop))',
							'Surface types_1 (Non-Stick/Ceramic (Cookware, Dishware, etc))',
							'Surface types_1 (Cast Iron (Cookware/Grills/etc))',
							'Surface types_1 (Porcelain (Dishware, premium crockery, sink, etc))',
							'Surface types_1 (Glass (Cookware/ Dishware))', 'Surface types_1 (Glass (Cooktop))',
							'Surface types_1 (Plastic (Dishware, Utensils, etc))',
							'Surface types_1 (Everyday dishware & utensils)',
							'Surface types_1 (Suitable for use across most surface types)',
							'Stain types_1 (Good at removing watermarks)',
							'Stain types_1 (Good at removing light (non-oily) stains)',
							'Stain types_1 (Good at removing tea/coffee stains)',
							'Stain types_1 (Good at removing light oil stains)',
							'Stain types_1 (Good at removing heavy oil stains)',
							'Stain types_1 (Good at removing sticky stains)',
							'Stain types_1 (Good at removing stubborn burnt stains)',
							'Stain types_1 (Effective at removing different stuck on and oily stains)',
							'Benefit_1 (Will not scratch or damage cleaning surfaces)', 'Benefit_1 (Durable/long lasting)',
							'Benefit_1 (Antibacterial/Prevent bacterial growth)',
							'Benefit_1 (Foam well with a small amount of detergent)', 'Benefit_1 (Easy to grip)',
							"Benefit_1 (Easy to rinse clean after use (food doesn't get stuck))",
							"Benefit_1 (Quick to dry after use (doesn't retain water))",
							'Benefit_1 (Easy to reach and clean tight corners and grooves)',
							'Benefit_1 (Antimicrobial/Prevent mold & odor)', 'Benefit_1 (Stain resistant)',
							'Claim_1 (Make cleaning a breeze, saves time to do the things you love )',
							'Claim_1 (Transform cleaning into a fun experience with foamy bubble)',
							'Claim_1 (Create a shiny finish after use)', 'Claim_1 (Guarantee satisfaction after use)',
							'Claim_1 (Comfortable to use )', 'Claim_1 (Eco-friendly/Good for the environment)',
							'Claim_1 (Keeps your family safe)', 'Claim_1 (Fun designs/ shapes to elevate your mood)',
							'Claim_1 (Scented (emits pleasant scent while cleaning))',
							'Claim_1 (Convenient as no soap required (infused with  cleaning detergent))',
							'Material_1 (Metal (e.g. Stainless Steel))', 'Material_1 (Man-made fibre (e.g., polyester))',
							'Material_1 (100% Recycled PET plastic)', 'Material_1 (Melamine foam (magic eraser))',
							'Material_1 (Natural corn based fibre)', 'Material_1 (Natural coconut based fibre)',
							'Material_1 (Natural cellulose sponge)',
							'Material_1 (Flexible foam (texture changes with water temperature))',
							'Material_1 (Stay Fresh foam (foam that resist stains and odors))', 'Price_1',
							'Color_1 (Green)', 'Color_1 (Dark Green)', 'Color_1 (Yellow)', 'Color_1 (Orange)',
							'Color_1 (Pink)', 'Color_1 (Blue)', 'Color_1 (Purple)', 'Color_1 (Beige/Brown )',
							'Color_1 (Black )', 'Color_1 (White)', 'Color_1 (Grey)', 'Shape_1 (Rectangle)',
							'Shape_1 (Wave )', 'Shape_1 (Tear drop )', 'Shape_1 (Round)', 'Shape_1 (Leaf)',
							'Shape_1 (Fish -shape)', 'Shape_1 (Square)', 'Shape_1 (Oval )', 'Shape_1 (Smiley)',
							'Shape_1 (Flower)']
		Full_name_list = []
		for ele in Output_Full_Name:
			temp1, temp2 = split_brand_string(ele)
			if temp1 != 'Format':
				Full_name_list.append([temp1, temp2])


	Att_list = [x[0] for x in Full_name_list]
	Label_list = [x[1] for x in Full_name_list]


	if Add_Constant:
		Full_name_df = pd.DataFrame({'Alternative' : Alt_labels, 'Attribute': ['Intercept'] * nc, 'Parameter': ['Intercept'] * nc})
	if Alternative_Specific_Config:
		for label in Alt_labels:
			temp = pd.DataFrame({'Alternative': [label] * len(Att_list), 'Attribute': Att_list, 'Parameter': Label_list})
			Full_name_df = pd.concat([Full_name_df,temp], ignore_index=True)
	else:
		Full_name_df = pd.DataFrame(Full_name_list, columns=['Attribute', 'Parameter'])

	if Add_Demo_Variables:
		Demo_list = []
		for ele in Ut_Demo_Variables:
			temp1, temp2 = split_brand_string(ele)
			Demo_list.append([temp1, temp2])

		Demo_Att_list = [x[0] for x in Demo_list]
		Demo_Label_list = [x[1] for x in Demo_list]

		if Add_Demo_Alternative_Specific_Config:
			for label in Alt_labels:
				temp = pd.DataFrame({'Alternative': [label] * len(Demo_Att_list), 'Attribute': Demo_Att_list, 'Parameter': Demo_Label_list})
				Full_name_df = pd.concat([Full_name_df,temp], ignore_index=True)
		else:
			if Alternative_Specific_Config:
				temp = pd.DataFrame({'Alternative': ['Generic'] * len(Demo_Att_list), 'Attribute': Demo_Att_list, 'Parameter': Demo_Label_list})
			else:
				temp = pd.DataFrame({'Attribute': Demo_Att_list, 'Parameter': Demo_Label_list})
			Full_name_df = pd.concat([Full_name_df, temp], ignore_index=True)

	if Model_Config != 'Labeled':
		Output_Est_Name = All_UT_Spec[0]
		Est_name_list = []
		for ele in Output_Est_Name:
			temp1, temp2 = split_brand_string(ele)
			Est_name_list.append([temp1, temp2])
	else:
		Est_name_list = []
		if Add_Constant:
			for i in range(1,nc):
				Est_name_list.append([Alt_labels[i],'Intercept','Intercept'])

		exclude_list = ['sero', 'uno'] + Ut_Demo_Variables
		if Alternative_Specific_Config:
			for index, labe_list in enumerate(All_UT_Spec):
				for ele in labe_list:
					if ele not in exclude_list:
						temp1, temp2 = split_brand_string(ele)
						Est_name_list.append([Alt_labels[index],temp1, temp2])
		else:
			Output_Est_Name = All_UT_Spec[0]
			Est_name_list = []
			for ele in Output_Est_Name:
				if ele not in exclude_list:
					temp1, temp2 = split_brand_string(ele)
					Est_name_list.append([temp1, temp2])

		if Add_Demo_Variables:
			Demo_list = []
			for ele in Ut_Demo_Variables:
				temp1, temp2 = split_brand_string(ele)
				Demo_list.append([temp1, temp2])

			Demo_Att_list   = [x[0] for x in Demo_list]
			Demo_Label_list = [x[1] for x in Demo_list]

			if Add_Demo_Alternative_Specific_Config:
				for i in range(1,nc):
					for temp1, temp2 in zip(Demo_Att_list, Demo_Label_list):
						Est_name_list.append([Alt_labels[i], temp1, temp2])
			else:
				if Alternative_Specific_Config:
					for temp1, temp2 in zip(Demo_Att_list, Demo_Label_list):
						Est_name_list.append(['Generic', temp1, temp2])
				else:
					for temp1, temp2 in zip(Demo_Att_list, Demo_Label_list):
						Est_name_list.append([temp1, temp2])


	uni_pid = Main_data['ID'].unique()
	print(f'Number of threads       : {Num_Threads}')
	print(f'Number of respondents   : {int(uni_pid.shape[0])}')
	print('-------------------------------------------')

	# *****************************************************************************
	#                 Starting Parameter values
	# *****************************************************************************
	dgp_beta_rum = External_Param[0:nvarma]
	if Model_Type == 'MNP':
		# Covariance matrix for the differenced error-covariance term vector. Only specify the lower half as it is symmetric
		nCholErr = int((nc) * (nc-1) * 0.5)
		dgp_Psi = External_Param[nvarma:nvarma + nCholErr]
		dgp_Psi = xpnd(dgp_Psi)

		row_psi = dgp_Psi.shape[0]

		psi_active = np.array([[0,
									1, 1,
									0, 0, 1,
									1, 0, 1, 1,
									0, 0, 1, 1, 1,
									0, 1, 1, 0, 0, 1,
									1, 0, 0, 1, 1, 0, 1,
									0, 1, 1, 0, 1, 1, 1, 1,
									1, 0, 0, 1, 1, 1, 0, 0, 1,
									0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
									1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]]).T


	# ****************************************************************************************************************************************************************************************
	#                 Packing of all parameters in a single vector (Do not change anything below this line)
	# *****************************************************************************************************************************************************************************************
	dgp_bd1 = dgp_beta_rum
	if Model_Type == 'MNP':
		dgp_Psi1 = dgp_Psi

	dgp_X1 = dgp_bd1
	if Model_Type == 'MNP':
		dgp_X1 = np.vstack((dgp_X1, vech(dgp_Psi1)))

	bb = dgp_beta_rum
	if Model_Type == 'MNP':
		bb = np.vstack((bb, vech(dgp_Psi1)))

	# Splitting data based on number of threads such that calculations can happen in parallel
	Data_Split = np.zeros((Num_Threads, 2))
	for i in range(1, Num_Threads + 1, 1):
		Data_Split[i - 1, 0] = int(ceil((i - 1) * ((nind - 1) / Num_Threads)) + 1)
		if (i != Num_Threads):
			Data_Split[i - 1, 1] = int(ceil(i * ((nind - 1) / Num_Threads)))
		else:
			Data_Split[i - 1, 1] = nind

	Data_Split = Data_Split - 1
	print("Share Calculation has Started.............")
	MNP_lpr   = lpr(bb,Main_data,dp_progress)
	Av_Share  = (np.mean(MNP_lpr, axis=col_wise))
	Std_Share = (np.std(MNP_lpr, axis=col_wise))

	Av_Share  = 100*Av_Share
	Std_Share = 100*Std_Share

	Av_Share = np.round(Av_Share, 1)
	Std_Share = np.round(Std_Share, 1)

	Summary_table = pd.DataFrame({'Alternatives': Alt_labels, 'Share (%)': Av_Share, 'Std. Dev. (%)': Std_Share})
	return Summary_table

def run():
	st.session_state.run = True

def unable_buttons():
	st.session_state.run  = False
	return st.rerun()


def main():
	if 'run' not in st.session_state:
		st.session_state.run = False
	table1, table2= st.columns([1.0, 2])
	with table1:
		html_str = f"""<style>p.a{{font-size:26px;font-family:sans serif;color:blue; text-align:center}}</style><p class="a">3M Korea Scenario Share Calculator</p>"""
		st.markdown(html_str,unsafe_allow_html=True)
		uploaded_file = st.file_uploader("Upload an Excel (.xlsx) file", type=["xlsx"])
		if st.button("Calculate Share",type='primary',on_click=run, disabled=st.session_state.run):
			if uploaded_file is not None:
				try:
					# Read the Excel file
					dp_progress = st.empty()
					message_formatted1 = '<p style="font-size:26px;font-family:sans serif;color:blue; text-align:center">Reading Uploaded file</p>'
					dp_progress.markdown(message_formatted1, unsafe_allow_html=True)
					Scenario_df = pd.read_excel(uploaded_file,sheet_name='Data')
					Mapping_df = pd.read_excel(uploaded_file, sheet_name='Mapping')
					Shares = Calculate_Share(Scenario_df, Mapping_df,dp_progress)
					st.session_state["Item_table"] = Shares
					with table2:
						st.write('<p style="font-size:26px;font-family:sans serif;color:blue; text-align:left">   Market Share</p>',unsafe_allow_html=True)
						st.data_editor(st.session_state["Item_table"], hide_index=True, use_container_width=False,disabled=['Alternatives', 'Share (%)', 'Std. Dev. (%)'],
									   height=(470), column_config={'Alternatives' : {"alignment": "left", 'width': 200},
																	'Share (%)'    : {"alignment": "left", 'width': 70},
																	'Std. Dev. (%)': {"alignment": "left", 'width': 90}})
						unable_buttons()
				except Exception as e:
					st.error(f"Error reading the file: {e}")
					unable_buttons()

			else:
				st.warning("Please upload a file before clicking 'Process File'.")
				unable_buttons()

		if "Item_table" in st.session_state:
			with table2:
				st.write('<p style="font-size:26px;font-family:sans serif;color:blue; text-align:left">   Market Share</p>',unsafe_allow_html=True)
				st.data_editor(st.session_state["Item_table"], hide_index=True, use_container_width=False,
							   disabled=['Alternatives', 'Share (%)', 'Std. Dev. (%)'],
							   height=(470), column_config={'Alternatives': {"alignment": "left", 'width': 200},
															'Share (%)': {"alignment": "left", 'width': 70},
															'Std. Dev. (%)': {"alignment": "left", 'width': 90}})

if __name__ == "__main__":
	main()
