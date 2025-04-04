# vim: tabstop=4 shiftwidth=4 expandtab
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

__generated_with = "0.12.4"
app = marimo.App(
    width="full",
    app_title="MA0218 Mini Project: The Climate Forum",
    layout_file="layouts/main.slides.json",
)


@app.cell
def _():
    # Import all the required libraries
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sb
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
    from sklearn.svm import SVR, LinearSVR
    from sklearn.tree import DecisionTreeRegressor

    # Set the seaborn style
    sb.set_theme()
    return (
        BayesianRidge,
        IterativeImputer,
        SVR,
        enable_iterative_imputer,
        mo,
        np,
        pd,
        plt,
        re,
        sb,
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
    # The constants used in the program
    DATA_FILE = "./data.xls"

    # The list of columns to drop
    COLUMNS_TO_DROP = [
        # Drop the country code as we are using the country name
        "Country code",
        # These two columns below are useless
        "SCALE",
        "Decimals",
        # There is no data for 2011, so just drop it
        "2011",
    ]

    # The list of series codes for problem 1
    SERIES_CODES_PROBLEM_1 = [
        # CO2 emissions per capita (metric tons)
        "EN.ATM.CO2E.PC",
        # CO2 emissions per units of GDP (kg/$1,000 of 2005 PPP $)
        "EN.ATM.CO2E.PP.GD.KD",
        # CO2 emissions, total (KtCO2)
        "EN.ATM.CO2E.KT",
        # Methane (CH4) emissions, total (KtCO2e)
        "EN.ATM.METH.KT.CE",
        # Nitrous oxide (N2O) emissions, total (KtCO2e)
        "EN.ATM.NOXE.KT.CE",
        # Other GHG emissions, total (KtCO2e)
        "EN.ATM.GHGO.KT.CE",
        # Energy use per capita (kilograms of oil equivalent)
        "EG.USE.PCAP.KG.OE",
        # Energy use per units of GDP (kg oil eq./$1,000 of 2005 PPP $)
        "EG.USE.COMM.GD.PP.KD",
    ]

    # The list of all the series codes needed
    SERIES_CODES_PROBLEM_2 = [
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
        #
        # Data every 5 years from 1990 - 2005
        #
        # Other GHG emissions, total (KtCO2e)
        "EN.ATM.GHGO.KT.CE",
        # Methane (CH4) emissions, total (KtCO2e)
        "EN.ATM.METH.KT.CE",
        # Nitrous oxide (N2O) emissions, total (KtCO2e)
        "EN.ATM.NOXE.KT.CE",
        #
        # Data every 5 years from 1990 - 2005 plus 2008
        #
        # Access to improved water source (% of total pop.)
        "SH.H2O.SAFE.ZS",
        # Access to improved sanitation (% of total pop.)
        "SH.STA.ACSN",
    ]

    # The list of regions for problem 1
    REGIONS_FOR_PROBLEM_1 = [
        "East Asia & Pacific",
        "Europe & Central Asia",
        "Euro area",
        "Latin America & Caribbean",
        "Middle East & North Africa",
        "South Asia",
        "Sub-Saharan Africa",
        "Small island developing states",
        "World",
    ]

    # The list of regions to remove for problem 2
    REGIONS_TO_REMOVE_FOR_PROBLEM_2 = [
        "East Asia & Pacific",
        "Europe & Central Asia",
        "Euro area",
        "High income",
        "Latin America & Caribbean",
        "Low income",
        "Lower middle income",
        "Low & middle income",
        "Middle income",
        "Middle East & North Africa",
        "South Asia",
        "Sub-Saharan Africa",
        "Upper middle income",
        "Small island developing states",
        "World",
    ]

    # The range of years in the data set.
    #
    # There is no data for 2011, so we are skipping it
    YEAR_RANGE = [str(year) for year in range(1990, 2010 + 1)]
    return (
        COLUMNS_TO_DROP,
        DATA_FILE,
        REGIONS_FOR_PROBLEM_1,
        REGIONS_TO_REMOVE_FOR_PROBLEM_2,
        SERIES_CODES_PROBLEM_1,
        SERIES_CODES_PROBLEM_2,
        YEAR_RANGE,
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
    SERIES_CODES_PROBLEM_2,
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
        "Function to clean up the data."

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
                ~cleaned_data["Series code"].isin(SERIES_CODES_PROBLEM_2)
            ].index,
            inplace=True,
        )

        # Reset the index of the data frame
        cleaned_data.reset_index(drop=True, inplace=True)

        # Return the data
        return cleaned_data

    # Create the function to remove everything after a line of code
    def strip_unnecessary_code(line_of_code: str) -> str:
        return re.sub(
            f"({line_of_code}).*?'", "\\1&quot;'", mo.show_code().text
        )

    # Save the code to clean the data for later
    clean_data_function_code = html(
        strip_unnecessary_code("return cleaned_data")
    )
    return clean_data, clean_data_function_code, strip_unnecessary_code


@app.cell
def _(
    BayesianRidge,
    IterativeImputer,
    YEAR_RANGE,
    html,
    pd,
    strip_unnecessary_code,
):
    # Create the function to impute the missing data
    def impute_missing_data(given_data: pd.DataFrame) -> pd.DataFrame:
        "Function to impute the missing data row by row."

        # Initialise the imputer object
        imputer_object = IterativeImputer(estimator=BayesianRidge())

        # Make a copy of the data
        imputed_data = given_data.copy()

        # Iterate over the given data
        for index, row in given_data[YEAR_RANGE].iterrows():
            #

            # Impute the data for the row
            imputed_row = imputer_object.fit_transform(
                list(zip(YEAR_RANGE, row))
            )

            # Remove the year range from the imputed row
            imputed_row_data = [value for (_, value) in imputed_row]

            # Set the imputed row data to the imputed data
            imputed_data.loc[index, YEAR_RANGE] = imputed_row_data

        # Return the imputed data
        return imputed_data

    # Save the code to impute the data for later
    impute_missing_data_function_code = html(
        strip_unnecessary_code("return imputed_data")
    )
    return impute_missing_data, impute_missing_data_function_code


@app.cell
def _(clean_data, data):
    # Clean the data
    cleaned_data = clean_data(data)

    cleaned_data
    return (cleaned_data,)


@app.cell
def _(cleaned_data, impute_missing_data):
    # Impute the missing data
    imputed_data = impute_missing_data(cleaned_data)

    imputed_data
    return (imputed_data,)


@app.cell
def _(pd):
    def format_data_for_problem(
        given_data: pd.DataFrame,
        regions: list[str],
        series_codes: list[str],
    ) -> dict[str, pd.DataFrame]:
        "Function to format the data properly for the problems."

        # Initialise the dictionary to store the data for problem 1
        formatted_data = {}

        # Iterate over all the regions required for problem 1
        for region in regions:
            #

            # Get the region data
            region_data = given_data.loc[given_data["Country name"] == region]

            # Grab the data that is in the series for problem 1
            region_series_data = (
                region_data.loc[region_data["Series code"].isin(series_codes)]
                .drop(columns=["Country name", "Series code"])
                .reset_index(drop=True)
            )

            # Pivot the data so that the series name is at the top
            pivoted_region_data = region_series_data.pivot_table(
                columns="Series name"
            )

            # Add the data to the dictionary
            formatted_data[region] = pivoted_region_data

        # Return the dictionary containing the formatted data
        return formatted_data

    return (format_data_for_problem,)


@app.cell
def _(create_problem_1_data, imputed_data, pd):
    # Create the data for problem 1
    problem_1_data = create_problem_1_data(imputed_data)
    # import linreg
    from sklearn.linear_model import LinearRegression

    # create model object
    model = LinearRegression()

    # prepare data for testing
    timeline = pd.DataFrame(YEAR_RANGE)
    world_data = problem_1_data["World"]
    var = [
        "CO2 emissions per capita (metric tons)",
        "CO2 emissions per units of GDP (kg/$1,000 of 2005 PPP $)",
        "CO2 emissions, total (KtCO2)",
        "GHG net emissions/removals by LUCF (MtCO2e)",
        "Methane (CH4) emissions, total (KtCO2e)",
        "Nitrous oxide (N2O) emissions, total (KtCO2e)",
        "Other GHG emissions, total (KtCO2e)",
        "Energy use per capita (kilograms of oil equivalent)",
        "Energy use per units of GDP (kg oil eq./$1,000 of 2005 PPP $)",
    ]

    world_data_c02 = pd.DataFrame(
        world_data["CO2 emissions per capita (metric tons)"]
    )
    # timeline_train = pd.DataFrame(timeline[:16])
    # world_data_c02_train = pd.DataFrame(world_data_c02[:16])
    # timeline_test = pd.DataFrame(timeline[-5:])
    # world_data_c02_test = pd.DataFrame(world_data_c02[-5:])

    # train linreg
    # model.fit(timeline, world_data_c02)
    # regline_x = timeline
    # regline_y = model.predict(regline_x)

    # # visualise the data regression
    # f = plt.figure()
    # plt.scatter(timeline,world_data_c02)
    # plt.plot(regline_x, regline_y, 'r-', linewidth = 3)
    # plt.show()
    return (
        LinearRegression,
        model,
        problem_1_data,
        timeline,
        var,
        world_data,
        world_data_c02,
    )


@app.cell
def _(problem_1_data):
    problem_1_data
    return


@app.cell
def _(model, plt, timeline, world_data):
    count = 0
    f, axes = plt.subplots(1, 8, figsize=(18, 30))
    for i in world_data:
        data1 = world_data[i]
        # train linreg
        model.fit(timeline, data1)
        regline_x = timeline
        regline_y = model.predict(regline_x)

        # visualise the data regression
        axes[count].scatter(timeline, data1)
        axes[count].plot(regline_x, regline_y, "r-", linewidth=3)
        count += 1

    plt.show()
    return axes, count, data1, f, i, regline_x, regline_y


@app.cell
def _(problem_1_data):
    problem_1_data
    return


@app.cell
def _(model, np, timeline_train, world_data_c02_train):
    # Checking goodness of fit
    # Explained Variance (R^2)
    print(
        "Explained Variance (R^2) \t:",
        model.score(timeline_train, world_data_c02_train),
    )

    # Mean Squared Error (MSE)
    def mean_sq_err(actual, predicted):
        """Returns the Mean Squared Error of actual and predicted values"""
        return np.mean(np.square(np.array(actual) - np.array(predicted)))

    mse = mean_sq_err(world_data_c02_train, model.predict(world_data_c02_train))
    print("Mean Squared Error (MSE) \t:", mse)
    print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mse))
    return mean_sq_err, mse


@app.cell
def _():
    return


@app.cell
def _(
    REGIONS_FOR_PROBLEM_1,
    SERIES_CODES_PROBLEM_1,
    format_data_for_problem,
    imputed_data,
):
    # Create the data for problem 1
    problem_1_data = format_data_for_problem(
        imputed_data, REGIONS_FOR_PROBLEM_1, SERIES_CODES_PROBLEM_1
    )
    problem_1_data
    return (problem_1_data,)


@app.cell
def _(
    REGIONS_TO_REMOVE_FOR_PROBLEM_2,
    SERIES_CODES_PROBLEM_2,
    format_data_for_problem,
    imputed_data,
):
    problem_2_data = format_data_for_problem(
        imputed_data,
        imputed_data[
            ~imputed_data["Country name"].isin(REGIONS_TO_REMOVE_FOR_PROBLEM_2)
        ]["Country name"].unique(),
        SERIES_CODES_PROBLEM_2,
    )

    problem_2_data
    return (problem_2_data,)


@app.cell
def _(SVR, pd, problem_2_data):
    svr_model = LinearSVR()
    linear_model = Ridge()
    decision_tree_model = DecisionTreeRegressor(max_depth=2)
    years = np.array(list(range(1990, 2006)))
    years_str = [str(year) for year in years]
    years = years.reshape(-1, 1)
    co2_data = pd.DataFrame(problem_2_data["Singapore"]["GDP ($)"])
    train_data = np.array(co2_data.loc[years_str]).ravel()

    svr_model.fit(years, train_data)
    linear_model.fit(years, train_data)
    decision_tree_model.fit(years, train_data)

    new_years = np.array(list(range(1990, 2011))).reshape(-1, 1)
    prediction = linear_model.predict(new_years)
    plt.figure(figsize=(16, 8))
    plt.plot(years, train_data)
    plt.plot(new_years, prediction, "r-", linewidth=3)
    return co2_data, svr_model


if __name__ == "__main__":
    app.run()
