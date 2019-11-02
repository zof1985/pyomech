# coding: utf-8


def polynome_txt(fit, magnitude_threshold=2, digits=3):
    """
    convert a fitting polynome to its mathematical string

    Input:
        fit: (ndarray, list)
            a list of float values defining the coefficients.

        magnitude_threshold: (float)
            the threshold that has not to be passed in order to avoid the use of the scientific notation
            for the representation of each coefficient.

        digits: (int)
            the decimal precision to be used for each coefficient

    Output:
        txt: (str)
            a string in the form: Y = aX^n + bX^n-1 ....
    """

    # import the dependancies
    import numpy as np

    # check the entries
    classcheck(fit, ['list', 'ndarray'])
    classcheck(magnitude_threshold, ['int', 'float'])
    classcheck(digits, ['int'])

    # ensure fit is a 1D array
    fit = np.atleast_1d(fit)
    assert fit.ndim == 1, "'fit' must have 1 dimension."

    # get the formatter
    formatter = "{:0." + str(digits) + "f}"

    # get the output string
    txt = r"$Y="
    for i, v in enumerate(fit):

        # add the sign
        sign = "+" if v >= 0 else "-"
        v = abs(v)

        # add the coefficient
        mag =  magnitude(float(v))
        if mag > 10 ** magnitude_threshold or v < 10 ** -magnitude_threshold:
            v_txt = formatter.format(v / (10 ** mag)) + "\cdot" + str(10) + "^{" + str(mag) + "}"
        else:
            v_txt = formatter.format(v)

        # agg the X in the correct order
        order = len(fit) - i - 1
        x_txt = ""
        if order > 0:
            x_txt += "\ X"
        if order > 1:
            x_txt += "^{" + str(order) + "}"

        # aggregate all
        txt += sign + v_txt + x_txt
    txt += "$"

    # return the text
    return txt


def classcheck(obj, classes, txt=None):
	"""
	Check if obj is any of the classes provided. If not an error is thrown with message txt.

	Input:
		obj: (object)
			the obkect to be tested.
		classes: (list)
			a list of str reflecting the names of the classes to be tested.
		txt: (None or str)
			the error message to be thrown in case of unmatch of the obj class with classes. If None, an automatic
			message is thrown.
	"""

	# check classes
	assert classes.__class__.__name__ in ["list", "ndarray"], "'classes' must be a list."

	# check txt
	if txt is None: txt = "'obj' must be any of " + str(classes) + "."
	assert txt.__class__.__name__ == "str", "'txt' must be a str."

	# perform the test
	assert any([obj.__class__.__name__[:len(i)] == i for i in classes]), txt



def magnitude(value, base=10):
    """
    return the order in the given base of the value

    Input
        value: (float)
            the value to be checked
        base: (float)
            the base to be used to define the order of the number

    Output
        mag: (float)
            the number required to elevate the base to get the value
    """

    # import the necessary packages
    from numpy import floor, log

    # check the entered values
    classcheck(value, ["int", "float"])
    classcheck(base, ["int", "float"])

	# return the magnitude
    if value == 0 or base == 0:
        return 0
    else:
        return int(floor(log(abs(value)) / log(base)))



def get_files(path, extension=None, check_subfolders=False):
    """
    list all the files having the required extension in the provided folder and its subfolders (if required).

    Input:
        path: (str)
            a directory where to look for the files.
        extension: (str)
            a str object defining the ending of the files that have to be listed.
        check_subfolders: (bool)
            if True, also the subfolders found in path are searched, otherwise only path is checked.

    Output:
        files: (list)
            a list containing the full_path to all the files corresponding to the input criteria.
    """

    # check entered data
    classcheck(path, ["str"])
    classcheck(extension, ["NoneType", "str"])
    classcheck(check_subfolders, ["bool"])

    # import dependancies
    import os

	# if path is not a folder return an empty list
    if os.path.isfile(path): return []

    # set the number of characters that have to be checked as extension
    n = None if extension is None else len(extension)

	# search within path and its subfolders
    try:
        files = [path + "\\" + i for i in os.listdir(path)]
    except FileNotFoundError:
        files = []
    i = 0
    while i < len(files):

        # if the current element is a folder update the list according to check_subfolders
        if os.path.isdir(files[i]):
            if check_subfolders: files += [files[i] + "\\" + j for j in os.listdir(files[i])]
            files.pop(i)

		# if the current element is a file with different extension as the one provided remove it from the list
        elif extension is not None and files[i][-n:] != extension: files.pop(i)

		# keep the file and move forward
        else: i += 1

	# return the list
    return files



def get_path(file):
	"""
	function to get the full path of a file
	"""

	# import the required packages
	import os

	# if the file is empty, return an empty string
	if not file: return file

	# if only the filename is provided,
	# add the current working directory path
	file = file.replace("/", "\\")
	out = (os.getcwd() + "\\") if len(file.split("\\")) < 2 else "" + file

	# check if the file contains " elements and remove them
	out = out[1:] if out[0] == "" else out
	out = out[:-1] if out[-1] == "" else out
	out = out[1:] if out[0] == "" else out
	out = out[:-1] if out[-1] == "" else out

	# ensure that the drive letter is capital
	out = out[0].upper() + out[1:]

	# check if the output file is a file or a directory
	out += "\\" if os.path.isdir(out) and out[-1:] != "\\" else ""

	# return the updated string
	return out



def lvlup(file, sep="\\"):
	"""
	Goes to the superior level in the directory path.

	Input:
		fil: (str)
			a file or a directory. Otherwise a message is casted and the function returns the input as is
		sep: (str)
			the characters defining the directory separator

	Output:
		a string reflecting the superior directory of file.
	"""

	# import the required packages
	from os.path import exists

	# check whether file exists as "file" or directory
	if not exists(str(file)): return file

	up = str(file).split(sep)[:-1]
	up = up[:-1] if file.endswith(sep) else up
	if len(up) == 0:

		# No valid level have been found over file, thus return file
		return file
	else:
		out = ""
		for s in up:
			out += s + sep
		return out



def to_excel(P, D, N="Sheet1", new_file=False):
	"""
	a shorthand function to save a pandas dataframe to an excel file

	Input
		P: (str)
			the path to the file where to store the file.

		D: (DataFrame)
			a dataframe.

		new_file: (boolean)
			if True, a completely new file will be created.

	Output
		The functions stores the data to the indicated file.
	"""

	# import the necessary packages
	from openpyxl import Workbook, load_workbook
	from os.path import exists

	# check files
	classcheck(P, ["str"])
	assert P.split(".")[-1] == "xlsx", "'P' must end with '.xlsx'"
	classcheck(D, ["DataFrame"])
	assert D.shape[0] <= 1048576, "'D' has more than 1048576 rows."
	assert D.shape[1] <= 16384, "'D' has more than 16384 columns."
	classcheck(N, ["str"])
	classcheck(new_file, ["bool"])

	# get the workbook
	if exists(P) and not new_file:
		wb = load_workbook(P)
	else:
		wb = Workbook()
		try:
			sh = wb["Sheet"]
			wb.remove(sh)
		except Exception:
			pass

	# get the sheet
	try:
		sh = wb[N]
		wb.remove(sh)
	except Exception:
		pass
	sh = wb.create_sheet(N)

	# update the data
	V = D.values
	[R, C] = V.shape

	# write the headers
	[sh.cell(1, i + 1, v) for i, v in enumerate(D.columns.tolist())]

	# write the data
	[sh.cell(r + 2, c + 1, V[r, c]) for r in range(R) for c in range(C)]

	# save data
	wb.save(P)



def from_excel(path, sheets=None, verbose=False):
	"""
	a shorthand function to collect data from an excel file and to store them into a dict.

	Input
		path: (str)
			the path to the file where to store the file.
		sheets: (list of str)
			the name of the sheets to be imported. If None all sheets will be imported.

	Output
		a dict object with keys equal to the sheets name and pandas dataframe as elements of each sheet
		in the excel file.
	"""

	# import the required packages
	from pandas import ExcelFile, read_excel
	from numpy import array

	# retrive the data in the path file
	try:
		xlfile = ExcelFile(path)
	except Exception:
		if verbose: print(path + " is not valid.")
		try: xlfile.close()
		except Exception: pass
		return None

	# get the sheets name
	sheets = array(xlfile.sheet_names if sheets is None else [sheets]).flatten()

	# get the data
	dfs = {i: read_excel(path, i) for i in sheets}

	# close the excel file
	xlfile.close()

	# return the dict
	return dfs



def get_time(tic=None, toc=None, as_string=True, compact=True):
	"""
	get the days, hours, minutes and seconds between the two times. If only tic is provided, it is considered as
	the lapsed time. If neither tic nor toc are provided, the function returns the current time as float

	Input (optional)
		tic: (int)
			an integer representing the starting time
		toc: (int)
			an integer indicating the stopping time
		as_string: (bool)
			should the output be returned as string?
		compact: (bool)
			if "as_string" is true, should the time be reported in a compact or in an extensive way?

	Output
		If nothing is provided, the function returns the current time. If only tic is provided, the function
		returns the time value from it to now. If both tic and toc are provided, the function returns the time
		difference between them.
	"""

	# import the necessary packages
	import numpy as np
	from time import time

	# check what to do
	if tic is None: return time()
	elif toc is None: tm = np.float(tic)
	else: tm = np.float(toc - tic)

	# convert the time value in days, hours, minutes, seconds and milliseconds
	d = int(np.floor(tm / 86400))
	tm -= (d * 86400)
	h = int(np.floor(tm / 3600))
	tm -= (h * 3600)
	m = int(np.floor(tm / 60))
	tm -= (m * 60)
	s = int(np.floor(tm))
	tm -= s
	ms = int(np.round(1000 * tm, 0))

	# report the calculated time
	if not as_string:
		return {"Days": [d], "Hours": [h], "Minutes": [m], "Seconds": [s], "Milliseconds": [ms]}
	else:
		st = "{:0>2d}".format(d) + (" Days - " if not compact else ":")
		st += "{:0>2d}".format(h)
		st += " Hours - " if not compact else ":"
		st += "{:0>2d}".format(m)
		st += " Minutes - " if not compact else ":"
		st += "{:0>2d}".format(s)
		st += " Seconds - " if not compact else ":"
		st += "{:0>3d}".format(ms)
		st += " Milliseconds" if not compact else ""
		return st



def extract(data, factors):
	"""
	Extract from data the names according to factors.

	Input:
		data: (DataFrame)
			a dataframe containing the parameters
		factors: (dict)
			a dict having the factors to be considered as keys and one row for each combination of factors to extract

	Output:
		a list where:
			0. the extracted dataframe having the columns of the factors and of the indicated names
			1. the index of data to corresponding to the extracted values
	"""

	# import the necessary packages
	from numpy import argwhere

	# if factors is provided as dataframe, convert it to  a dict object
	try: factors = factors.to_dict("list")
	except Exception: factors = factors

	# get the index of the required subset
	index = argwhere(data[[f for f in factors]].isin(factors).all(1)).flatten().astype(int)
	index = data.index.to_numpy()[index]
	cols = data.columns.tolist()

	# return the subset
	tdata = data.loc[index, cols]
	return tdata, tdata.index.to_numpy().flatten()
