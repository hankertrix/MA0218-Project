# vim: tabstop=4 shiftwidth=4 noexpandtab
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "pandas==2.2.3",
#     "seaborn==0.13.2",
#     "xlrd==2.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.28"
app = marimo.App(
	width="full",
	app_title="MA0218 Mini Project: The Climate Forum",
	layout_file="layouts/main.slides.json",
)


@app.cell
def _(mo):
	# Create a function to more easily create HTML
	def html(*args: str) -> mo.Html:
		return mo.Html("\n".join(args))

	# Create a function to more easily create markdown
	def md(*args: str) -> mo.md:
		return mo.md("\n".join(args))

	html(
		md("# MA0218 Mini Project:").center().text,
		html("<h1>The Climate Forum</h1>").center().style(color="#3CB034").text,
		md("By: Nicholas, Haziq, Dylan and Jun Feng")
		.center()
		.style(padding="5em")
		.text,
	)
	return html, md


@app.cell
def _():
	# Import all the required libraries
	import re

	import marimo as mo
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import seaborn as sb

	# Set the seaborn style
	sb.set_theme()
	return mo, np, pd, plt, re, sb


@app.cell
def _():
	# The constants used in the program
	DATA_FILE = "./data.xls"

	# The list of columns to drop
	COLUMNS_TO_DROP = [
		# Drop the country code as we are using the country name
		"Country code",
		# Drop the series name as we are using the series code
		"Series name",
		# These two columns below are useless
		"SCALE",
		"Decimals",
	]

	# The list of series codes for series that have data for most years
	SERIES_CODES_MOST_YEARS = [
		#
		# Mostly complete data
		#
		# Cereal yield (kg per hectare)
		"AG.YLD.CREL.KG",
		# Foreign direct investment, net inflows (% of GDP)
		"BX.KLT.DINV.WD.GD.ZS",
		# Energy use per units of GDP (kg oil eq./$1,000 of 2005 PPP $)
		"EG.USE.COMM.GD.PP.KD",
		# Energy use per capita (kilograms of oil equivalent)
		"EG.USE.PCAP.KG.OE",
		# CO2 emissions, total (KtCO2)
		"EN.ATM.CO2E.KT",
		# CO2 emissions per capita (metric tons)
		"EN.ATM.CO2E.PC",
		# CO2 emissions per units of GDP (kg/$1,000 of 2005 PPP $)
		"EN.ATM.CO2E.PP.GD.KD",
		# Nationally terrestrial protected areas (% of total land area)
		"ER.LND.PTLD.ZS",
		# GDP ($)
		"NY.GDP.MKTP.CD",
		# GNI per capita (Atlas $)
		"NY.GNP.PCAP.CD",
		# Under-five mortality rate (per 1,000)
		"SH.DYN.MORT",
		# Population growth (annual %)
		"SP.POP.GROW",
		# Population
		"SP.POP.TOTL",
		# Urban population growth (annual %)
		"SP.URB.GROW",
		# Urban population
		"SP.URB.TOTL",
		#
		# Spotty data
		#
		# Population in urban agglomerations >1million (%)
		"EN.URB.MCTY.TL.ZS",
		# Paved roads (% of total roads)
		"IS.ROD.PAVE.ZS",
		# Ratio of girls to boys in primary & secondary school (%)
		"SE.ENR.PRSC.FM.ZS",
		# Primary completion rate, total (% of relevant age group)
		"SE.PRM.CMPT.ZS",
		# Physicians (per 1,000 people)
		"SH.MED.PHYS.ZS",
	]

	# The list of series codes for series
	# that have data every 5 years from 1990 - 2005
	SERIES_CODES_EVERY_5_YEARS = [
		# Other GHG emissions, total (KtCO2e)
		"EN.ATM.GHGO.KT.CE",
		# Methane (CH4) emissions, total (KtCO2e)
		"EN.ATM.METH.KT.CE",
		# Nitrous oxide (N2O) emissions, total (KtCO2e)
		"EN.ATM.NOXE.KT.CE",
	]

	# The list of series codes for series
	# that have data every 5 years from 1990 - 2005, and 2008
	SERIES_CODES_EVERY_5_YEARS_PLUS_2008 = [
		# Access to improved water source (% of total pop.)
		"SH.H2O.SAFE.ZS",
		# Access to improved sanitation (% of total pop.)
		"SH.STA.ACSN",
	]

	# The range of years in the data set
	YEAR_RANGE = [str(year) for year in range(1990, 2011 + 1)]

	# The list of years every 5 years from 1990 - 2005
	YEAR_RANGE_EVERY_5_YEARS = [
		"1990",
		"1995",
		"2000",
		"2005",
	]

	# The list of years every 5 years from 1990 - 2005, and 2008
	YEAR_RANGE_EVERY_5_YEARS_PLUS_2008 = [
		"1990",
		"1995",
		"2000",
		"2005",
		"2008",
	]
	return (
		COLUMNS_TO_DROP,
		DATA_FILE,
		SERIES_CODES_EVERY_5_YEARS,
		SERIES_CODES_EVERY_5_YEARS_PLUS_2008,
		SERIES_CODES_MOST_YEARS,
		YEAR_RANGE,
		YEAR_RANGE_EVERY_5_YEARS,
		YEAR_RANGE_EVERY_5_YEARS_PLUS_2008,
	)


@app.cell
def _(DATA_FILE, html, md, mo, pd):
	# Read the data from the data file
	data = pd.read_excel(DATA_FILE)

	# Convert all the column names to string
	data.columns = data.columns.astype(str)

	# Display the slide contents
	html(
		md("## Data set").text,
		html(
			"<p>",
			"The data set used is the climate change data set.",
			"Have a look at the data set in the table below:",
			"</p>",
		)
		.style(padding="10px 0px")
		.text,
		mo.ui.table(data, page_size=200).text,
	)
	return (data,)


@app.cell
def _(
	COLUMNS_TO_DROP,
	SERIES_CODES_EVERY_5_YEARS,
	SERIES_CODES_EVERY_5_YEARS_PLUS_2008,
	SERIES_CODES_MOST_YEARS,
	YEAR_RANGE,
	data,
	html,
	mo,
	np,
	pd,
	re,
):
	# Create the function to clean the data
	def clean_data(given_data: pd.DataFrame) -> pd.DataFrame:
		# Make a copy of the data
		cleaned_data = given_data.copy()

		# Convert all the column names to string
		cleaned_data.columns = data.columns.astype(str)

		# Drop the columns that aren't needed
		cleaned_data.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

		# Coerce all the data in the columns for the years to numeric
		cleaned_data[YEAR_RANGE] = cleaned_data[YEAR_RANGE].apply(
			lambda elem: pd.to_numeric(elem, errors="coerce")
		)

		# Replace all infinity values with NaNs
		cleaned_data[YEAR_RANGE] = cleaned_data[YEAR_RANGE].replace(
			[np.inf, -np.inf], np.nan
		)

		# Drop all the rows in the years that have all their values as NaNs
		cleaned_data.drop(
			cleaned_data[cleaned_data[YEAR_RANGE].isna().all(axis=1)].index,
			inplace=True,
		)

		# Drop all the rows with series we don't need
		cleaned_data.drop(
			cleaned_data[
				~cleaned_data["Series code"].isin(
					SERIES_CODES_MOST_YEARS
					+ SERIES_CODES_EVERY_5_YEARS
					+ SERIES_CODES_EVERY_5_YEARS_PLUS_2008
				)
			].index,
			inplace=True,
		)

		# Reset the index of the data frame
		cleaned_data.reset_index(drop=True, inplace=True)

		# Return the data
		return cleaned_data

	# Save the code to clean the data for later
	clean_data_function_code = html(
		# The regular expression here is to remove everything
		# after the return statement of the clean_data function
		# so that these lines of code won't be displayed
		re.sub("(return cleaned_data).*?'", "\\1&quot;'", mo.show_code().text)
	)
	return clean_data, clean_data_function_code


if __name__ == "__main__":
	app.run()
