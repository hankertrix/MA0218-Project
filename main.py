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

__generated_with = "0.12.0"
app = marimo.App(width="medium", layout_file="layouts/main.slides.json")


@app.cell
def _():
    # Import all the required libraries
    import re
    from operator import itemgetter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sb
    from matplotlib.ticker import StrMethodFormatter
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.impute import IterativeImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import (
        BayesianRidge,
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        PassiveAggressiveRegressor,
        RANSACRegressor,
        Ridge,
        SGDRegressor,
        TheilSenRegressor,
    )
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR, LinearSVR
    from sklearn.tree import DecisionTreeRegressor

    # Set the seaborn style
    sb.set_theme()
    return (
        BayesianRidge,
        DecisionTreeRegressor,
        ElasticNet,
        ExtraTreesRegressor,
        GaussianProcessRegressor,
        HuberRegressor,
        IsotonicRegression,
        IterativeImputer,
        KNeighborsRegressor,
        Lasso,
        LinearRegression,
        LinearSVR,
        PassiveAggressiveRegressor,
        PolynomialFeatures,
        RANSACRegressor,
        RandomForestRegressor,
        Ridge,
        SGDRegressor,
        SVR,
        StrMethodFormatter,
        TheilSenRegressor,
        enable_iterative_imputer,
        itemgetter,
        make_pipeline,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        re,
        sb,
    )


@app.cell
def _(mo):
    # Create a function to more easily create HTML
    def html(*args):
        return mo.Html(
            "\n".join(
                [arg if isinstance(arg, str) else arg.text for arg in args]
            )
        )

    # Create a function to more easily create markdown
    def md(*args):
        return mo.md(
            "\n".join(
                [arg if isinstance(arg, str) else arg.text for arg in args]
            )
        )

    html(
        md("# MA0218 Mini Project:").center(),
        html("<h1>The Climate Forum</h1>").center().style(color="#3CB034"),
        md("By: Nicholas, Haziq, Dylan and Jun Feng")
        .center()
        .style(padding="5em"),
    )
    return html, md


@app.cell
def _(np):
    # The constants used in the program
    DATA_FILE = "./data.xls"

    # The table page size to use
    TABLE_PAGE_SIZE = 20

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

    # The map of the series code to its multiplier for problem 2
    SERIES_CODE_TO_MULTIPLIER_MAP = {
        # Access to improved sanitation (% of total pop.)
        "SH.STA.ACSN": 1,
        # Access to improved water source (% of total pop.)
        "SH.H2O.SAFE.ZS": 1,
        # CO2 emissions per capita (metric tons)
        "EN.ATM.CO2E.PC": -1,
        # CO2 emissions per units of GDP (kg/$1,000 of 2005 PPP $)
        "EN.ATM.CO2E.PP.GD.KD": -1,
        # CO2 emissions, total (KtCO2)
        "EN.ATM.CO2E.KT": -1,
        # Cereal yield (kg per hectare)
        "AG.YLD.CREL.KG": 1,
        # Foreign direct investment, net inflows (% of GDP)
        "BX.KLT.DINV.WD.GD.ZS": 1,
        # GDP ($)
        "NY.GDP.MKTP.CD": 1,
        # GNI per capita (Atlas $)
        "NY.GNP.PCAP.CD": 1,
        # Nationally terrestrial protected areas (% of total land area)
        "ER.LND.PTLD.ZS": 1,
        # Paved roads (% of total roads)
        "IS.ROD.PAVE.ZS": 1,
        # Physicians (per 1,000 people)
        "SH.MED.PHYS.ZS": 1,
        # Ratio of girls to boys in primary & secondary school (%)
        "SE.ENR.PRSC.FM.ZS": 1,
        # Under-five mortality rate (per 1,000)
        "SH.DYN.MORT": -1,
        # Energy use per capita (kilograms of oil equivalent)
        "EG.USE.PCAP.KG.OE": -1,
        # Energy use per units of GDP (kg oil eq./$1,000 of 2005 PPP $)
        "EG.USE.COMM.GD.PP.KD": -1,
        # Methane (CH4) emissions, total (KtCO2e)
        "EN.ATM.METH.KT.CE": -1,
        # Nitrous oxide (N2O) emissions, total (KtCO2e)
        "EN.ATM.NOXE.KT.CE": -1,
        # Other GHG emissions, total (KtCO2e)
        "EN.ATM.GHGO.KT.CE": -1,
    }

    # The list of all the series codes needed
    SERIES_CODES = [
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
    YEAR_RANGE = np.array(list(range(1990, 2010 + 1)))
    YEAR_RANGE_STR = [str(year) for year in YEAR_RANGE]
    return (
        COLUMNS_TO_DROP,
        DATA_FILE,
        REGIONS_FOR_PROBLEM_1,
        REGIONS_TO_REMOVE_FOR_PROBLEM_2,
        SERIES_CODES,
        SERIES_CODES_PROBLEM_1,
        SERIES_CODE_TO_MULTIPLIER_MAP,
        TABLE_PAGE_SIZE,
        YEAR_RANGE,
        YEAR_RANGE_STR,
    )


@app.cell
def _(DATA_FILE, pd):
    # Read the data from the data file
    climate_change_excel_sheet = pd.read_excel(DATA_FILE, sheet_name=None)

    # Get the data from the excel sheet
    data = climate_change_excel_sheet["Data"]
    country_descriptions = climate_change_excel_sheet["Country"]
    series_descriptions = climate_change_excel_sheet["Series"]

    # Convert all the column names to string
    data.columns = data.columns.astype(str)
    return (
        climate_change_excel_sheet,
        country_descriptions,
        data,
        series_descriptions,
    )


@app.cell
def _(series_descriptions):
    # Create the map from the series code to the series name
    SERIES_CODE_TO_NAME_MAP = dict(
        zip(
            series_descriptions["Series code"],
            series_descriptions["Series name"],
        )
    )

    # Create the map from the series name to the series code
    SERIES_NAME_TO_CODE_MAP = dict(
        zip(
            series_descriptions["Series name"],
            series_descriptions["Series code"],
        )
    )
    return SERIES_CODE_TO_NAME_MAP, SERIES_NAME_TO_CODE_MAP


@app.cell
def _(TABLE_PAGE_SIZE, data, html, md, mo):
    # Display the slide contents
    html(
        md("## Data set"),
        html(
            "<p>",
            "The data set used is the climate change data set.",
            "Have a look at the data set in the table below:",
            "</p>",
        ).style(padding="10px 0px"),
        mo.ui.table(data, selection=None, page_size=TABLE_PAGE_SIZE),
    )
    return


@app.cell
def _(
    COLUMNS_TO_DROP,
    SERIES_CODES,
    YEAR_RANGE_STR,
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
        cleaned_data[YEAR_RANGE_STR] = cleaned_data[YEAR_RANGE_STR].apply(
            lambda elem: pd.to_numeric(elem, errors="coerce")
        )

        # Replace all infinity values with NaNs
        cleaned_data[YEAR_RANGE_STR] = cleaned_data[YEAR_RANGE_STR].replace(
            [np.inf, -np.inf], np.nan
        )

        # Drop all the rows in the years that have all their values as NaNs
        cleaned_data.drop(
            cleaned_data[cleaned_data[YEAR_RANGE_STR].isna().all(axis=1)].index,
            inplace=True,
        )

        # Drop all the rows with series we don't need
        cleaned_data.drop(
            cleaned_data[~cleaned_data["Series code"].isin(SERIES_CODES)].index,
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
    IterativeImputer,
    LinearRegression,
    PolynomialFeatures,
    YEAR_RANGE_STR,
    html,
    make_pipeline,
    pd,
    strip_unnecessary_code,
):
    # Create the function to impute the missing data
    def impute_missing_data(given_data: pd.DataFrame) -> pd.DataFrame:
        "Function to impute the missing data row by row."

        # Initialise the imputer object
        imputer_object = IterativeImputer(
            estimator=make_pipeline(PolynomialFeatures(3), LinearRegression())
        )

        # Make a copy of the data
        imputed_data = given_data.copy()

        # Iterate over the given data
        for index, row in given_data[YEAR_RANGE_STR].iterrows():
            #

            # Impute the data for the row
            imputed_row = imputer_object.fit_transform(
                list(zip(YEAR_RANGE_STR, row))
            )

            # Remove the year range from the imputed row
            imputed_row_data = [value for (_, value) in imputed_row]

            # Set the imputed row data to the imputed data
            imputed_data.loc[index, YEAR_RANGE_STR] = imputed_row_data

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
def _(md):
    md("# Exploratory Analysis")
    return


@app.cell
def _(md):
    md(
        md("## Exploratory Analysis: Procedure").style(padding="10px 0px"),
        md(
            "Below are the steps we followed to explore the data:",
            "",
            "1. Run through the data in Excel and note that the data",
            "is time series with data of at most 20 years.",
            "",
            "2. With such a small number of data points,",
            "a list of regression models was collated from Scikit-Learn",
            "to assess the ability of machine learning models to fit",
            "the data properly.",
            "",
            "3. Write code to evaluate model performance",
            "on one series and compare the results.",
        ),
    )
    return


@app.cell
def _(
    BayesianRidge,
    DecisionTreeRegressor,
    ElasticNet,
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    HuberRegressor,
    IsotonicRegression,
    KNeighborsRegressor,
    Lasso,
    LinearRegression,
    LinearSVR,
    PolynomialFeatures,
    RANSACRegressor,
    RandomForestRegressor,
    Ridge,
    SGDRegressor,
    SVR,
    StrMethodFormatter,
    YEAR_RANGE,
    YEAR_RANGE_STR,
    cleaned_data,
    make_pipeline,
    mean_squared_error,
    np,
    pd,
    plt,
    r2_score,
):
    def exploratory_analysis():
        "Exploratory analysis of machine learning models on the data set."

        # The target country and the data set to regress over
        target_country = "Singapore"

        # The target series to regress over, which is
        # GDP ($)
        target_series = "NY.GDP.MKTP.CD"

        # Reshape the years to make it usable for training
        x_train = YEAR_RANGE.reshape(-1, 1)

        # The data to use
        data = cleaned_data[
            (cleaned_data["Country name"] == target_country)
            & (cleaned_data["Series code"] == target_series)
        ]

        # The number of plots in a column
        number_of_plot_columns = 5

        # The training data for the series
        y_train = data[YEAR_RANGE_STR].values.flatten()

        # Regression models
        regression_models = {
            "Linear Regressor": LinearRegression(),
            "Ridge Regressor": Ridge(),
            "Lasso Regressor": Lasso(),
            "Bayesian Ridge": BayesianRidge(),
            "Elastic Net": ElasticNet(),
            "Huber Regressor": HuberRegressor(),
            "RANSAC Regressor": RANSACRegressor(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, random_state=42
            ),
            "Extra Trees Regressor": ExtraTreesRegressor(
                n_estimators=100, random_state=42
            ),
            "SVR (RBF Kernel)": SVR(kernel="rbf"),
            "SVR (Linear Kernel)": SVR(kernel="linear"),
            "SVR (Poly Kernel)": SVR(kernel="poly"),
            "SVR (Sigmoid Kernel)": SVR(kernel="sigmoid"),
            "Linear SVR": LinearSVR(),
            "Stochastic Gradient Descent Regressor": SGDRegressor(),
            "Gaussian Process Regressor": GaussianProcessRegressor(),
            "Isotonic Regressor": IsotonicRegression(),
            #
            # Decision Tree Regressors
            "Decision Tree Regressor (max depth 1)": DecisionTreeRegressor(
                max_depth=1
            ),
            "Decision Tree Regressor (max depth 2)": DecisionTreeRegressor(
                max_depth=2
            ),
            "Decision Tree Regressor (max depth 3)": DecisionTreeRegressor(
                max_depth=3
            ),
            "Decision Tree Regressor (max depth 4)": DecisionTreeRegressor(
                max_depth=4
            ),
            "Decision Tree Regressor (max depth 5)": DecisionTreeRegressor(
                max_depth=5
            ),
            "Decision Tree Regressor (max depth 6)": DecisionTreeRegressor(
                max_depth=6
            ),
            "Decision Tree Regressor (max depth 7)": DecisionTreeRegressor(
                max_depth=7
            ),
            "Decision Tree Regressor (max depth 8)": DecisionTreeRegressor(
                max_depth=8
            ),
            "Decision Tree Regressor (max depth 9)": DecisionTreeRegressor(
                max_depth=9
            ),
            "Decision Tree Regressor (max depth 10)": DecisionTreeRegressor(
                max_depth=10
            ),
            #
            # KNN Regressors
            "KNN Regressor (1 neighbour)": KNeighborsRegressor(n_neighbors=1),
            "KNN Regressor (2 neighbours)": KNeighborsRegressor(n_neighbors=2),
            "KNN Regressor (3 neighbours)": KNeighborsRegressor(n_neighbors=3),
            "KNN Regressor (4 neighbours)": KNeighborsRegressor(n_neighbors=4),
            "KNN Regressor (5 neighbours)": KNeighborsRegressor(n_neighbors=5),
            "KNN Regressor (6 neighbours)": KNeighborsRegressor(n_neighbors=6),
            "KNN Regressor (7 neighbours)": KNeighborsRegressor(n_neighbors=7),
            "KNN Regressor (8 neighbours)": KNeighborsRegressor(n_neighbors=8),
            "KNN Regressor (9 neighbours)": KNeighborsRegressor(n_neighbors=9),
            "KNN Regressor (10 neighbours)": KNeighborsRegressor(
                n_neighbors=10
            ),
            #
            # Linear regressor with polynomial features
            "Polynomial Regressor (degree 1)": make_pipeline(
                PolynomialFeatures(1), LinearRegression()
            ),
            "Polynomial Regressor (degree 2)": make_pipeline(
                PolynomialFeatures(2), LinearRegression()
            ),
            "Polynomial Regressor (degree 3)": make_pipeline(
                PolynomialFeatures(3), LinearRegression()
            ),
            "Polynomial Regressor (degree 4)": make_pipeline(
                PolynomialFeatures(4), LinearRegression()
            ),
            "Polynomial Regressor (degree 5)": make_pipeline(
                PolynomialFeatures(5), LinearRegression()
            ),
            "Polynomial Regressor (degree 6)": make_pipeline(
                PolynomialFeatures(6), LinearRegression()
            ),
            "Polynomial Regressor (degree 7)": make_pipeline(
                PolynomialFeatures(7), LinearRegression()
            ),
            "Polynomial Regressor (degree 8)": make_pipeline(
                PolynomialFeatures(8), LinearRegression()
            ),
            "Polynomial Regressor (degree 9)": make_pipeline(
                PolynomialFeatures(9), LinearRegression()
            ),
            "Polynomial Regressor (degree 10)": make_pipeline(
                PolynomialFeatures(10), LinearRegression()
            ),
        }

        # Get the number of plot rows
        number_of_plot_rows = int(
            np.ceil(len(regression_models) / number_of_plot_columns)
        )

        # Create the figure and the subplots
        figure, axes = plt.subplots(
            number_of_plot_rows,
            number_of_plot_columns,
            figsize=[number_of_plot_rows * number_of_plot_columns] * 2,
        )

        # Initialise the results
        results = []

        # Iterate over the regression models
        for index, (name, model) in enumerate(regression_models.items()):
            #

            # Fit the model on the data
            model.fit(x_train, y_train)

            # Get the prediction from the model
            prediction = model.predict(x_train)

            # Get the R squared score
            r_squared = r2_score(y_train, prediction)

            # Get the mean squared error and root mean squared error
            mse = mean_squared_error(y_train, prediction)
            rmse = np.sqrt(mse)

            # Append metrics to the list
            results.append(
                {
                    "Model": name,
                    "R squared score": r_squared,
                    "MSE": mse,
                    "RMSE": rmse,
                }
            )

            # Get the axis of the subplot
            axis = axes.flat[index]

            # Plot the data and the prediction
            axis.plot(x_train, y_train)
            axis.plot(x_train, prediction)

            # Set the title to the model
            axis.set_title(name)

            # Use integers for the x axis
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        # Put the data into a data frame
        results_dataframe = (
            pd.DataFrame(results)
            .sort_values(by="R squared score", ascending=False)
            .reset_index(drop=True)
        )

        # Return the results dataframe
        return results_dataframe, figure

    # Run the exploratory analysis
    exploratory_analysis_table, exploratory_analysis_figure = (
        exploratory_analysis()
    )
    return (
        exploratory_analysis,
        exploratory_analysis_figure,
        exploratory_analysis_table,
    )


@app.cell
def _(TABLE_PAGE_SIZE, exploratory_analysis_table, html, md, mo):
    # Create the slide for the table
    html(
        md("## Exploratory Analysis: Tabulated Results")
        .center()
        .style(padding_top="10px", padding_bottom="20px"),
        mo.ui.table(
            exploratory_analysis_table,
            selection=None,
            page_size=TABLE_PAGE_SIZE,
        ),
    )
    return


@app.cell
def _(exploratory_analysis_figure, html, md, mo):
    # Create the slide for the plot of all the models
    mo.output.append(
        html(
            md("## Exploratory Analysis: Plots of Regression Models")
            .center()
            .style(padding="10px 0px"),
        )
    )
    mo.output.append(exploratory_analysis_figure)
    return


@app.cell
def _(html, md):
    md(
        "# Question 1:",
        html("<h1>Are we getting greener or more sustainable?</h1>"),
    )
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
    LinearRegression,
    StrMethodFormatter,
    YEAR_RANGE,
    np,
    plt,
    problem_1_data,
):
    def solve_question_1():
        "Function to solve question 1."

        # Initialise the list of figures and axes
        all_figures = []

        # Get the list of years for training
        years = YEAR_RANGE.reshape(-1, 1)

        # Initialise the model to show the trend
        model = LinearRegression()

        # Create the sub plots
        figure, axes = plt.subplots(
            len(problem_1_data),
            np.max([len(data.columns) for data in problem_1_data.values()]),
            figsize=(64, 64),
        )

        # Iterate over all the countries
        for country_index, (country, data) in enumerate(problem_1_data.items()):
            #

            # Iterate over all of the columns in the data
            for column_index, column in enumerate(data):
                #

                # Get the column data
                column_data = data[column]

                # Fit the models on the data
                model.fit(years, column_data)

                # Predict the values for the data
                prediction = model.predict(years)

                # Get the axis
                axis = axes[country_index, column_index]

                # Plot the data and the prediction
                axis.scatter(years, column_data)
                axis.plot(years, column_data)
                axis.plot(years, prediction)
                axis.set_title(country)
                axis.set_xlabel("Year")
                axis.set_ylabel(column)

                # Set the x axis format to integers
                axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

                # Add the figure to the lists
                all_figures.append(figure)

        # Return the figure
        return figure

    solve_question_1()
    return (solve_question_1,)


@app.cell
def _(html, md):
    md(
        "# Question 2:",
        html("<h1>Which country should I move to in the future?</h1>"),
    )
    return


@app.cell
def _(
    REGIONS_TO_REMOVE_FOR_PROBLEM_2,
    SERIES_CODES,
    format_data_for_problem,
    imputed_data,
):
    problem_2_data = format_data_for_problem(
        imputed_data,
        imputed_data[
            ~imputed_data["Country name"].isin(REGIONS_TO_REMOVE_FOR_PROBLEM_2)
        ]["Country name"].unique(),
        SERIES_CODES,
    )

    problem_2_data
    return (problem_2_data,)


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    YEAR_RANGE,
    make_pipeline,
    np,
    pd,
    problem_2_data,
):
    def get_predictions_for_question_2():
        "Function to get the predictions for 2025 for question 2."

        # Create the dictionary of predictions
        predictions = {}

        # Initialise the model to do the predictions
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

        # Fix the year range for training
        years = YEAR_RANGE.reshape(-1, 1)

        # Iterate over the countries in the problem 2 data set
        for country, country_data in problem_2_data.items():
            #

            # Create the dictionary to store the predictions
            # for each of the series for the country
            country_predictions = {}

            # Iterate over the series in the country data
            for series, series_data in country_data.items():
                #

                # Fit the model
                model.fit(years, series_data)

                # Get the predicted value for 2025
                predicted_value = model.predict(np.array(2025).reshape(-1, 1))

                # Store the prediction
                country_predictions[series] = predicted_value[0]

            # Append the country predictions to the list
            predictions[country] = country_predictions

        # Convert the predictions to a dataframe
        predictions_dataframe = pd.DataFrame(predictions)

        # Normalise the predictions
        normalised_predictions = (
            predictions_dataframe - predictions_dataframe.mean()
        ) / predictions_dataframe.std()

        # Transpose the predictions
        transposed_predictions = normalised_predictions.transpose()

        # Fill all the NaNs with zeros
        transposed_predictions.fillna(0, inplace=True)

        return transposed_predictions

    normalised_predictions_question_2 = get_predictions_for_question_2()
    normalised_predictions_question_2
    return get_predictions_for_question_2, normalised_predictions_question_2


@app.cell
def _(
    SERIES_CODE_TO_MULTIPLIER_MAP,
    SERIES_NAME_TO_CODE_MAP,
    itemgetter,
    normalised_predictions_question_2,
    pd,
):
    def get_quality_of_life_score_for_question_2():
        "Function to get the quality of life score for question 2."

        # Initialise the dictionary with the quality of life scores
        # for each country
        quality_of_life_scores = {}

        # Get the list of series codes that are wanted
        wanted_series_codes = list(SERIES_CODE_TO_MULTIPLIER_MAP.keys())

        # Make a copy of the normalised predictions
        normalised_predictions = normalised_predictions_question_2.copy()

        # Convert all the series names to a series code
        normalised_predictions.columns = [
            SERIES_NAME_TO_CODE_MAP[series_name]
            for series_name in normalised_predictions.columns
        ]

        # Get only the wanted series
        wanted_predictions = normalised_predictions[wanted_series_codes]

        # Iterate over each country
        for country, country_data in wanted_predictions.iterrows():
            #

            # Initialise the quality of life score
            qol_score = 0

            # Iterate over all the columns in the country data
            for index, value in enumerate(country_data):
                #

                # Get the series code for the column
                series_code = wanted_predictions.columns[index]

                # Get the multiplier for the series code
                multiplier = SERIES_CODE_TO_MULTIPLIER_MAP[series_code]

                # Multiply the value by the multiplier
                # and add the result to the score
                qol_score += value * multiplier

            # Add the quality of life score for the country to the dictionary
            quality_of_life_scores[country] = qol_score

        # Sort the dictionary to have the countries
        # with the highest scores appear first
        sorted_quality_of_life_scores = dict(
            sorted(
                quality_of_life_scores.items(), key=itemgetter(1), reverse=True
            )
        )

        # Get the data frame for the quality of life scores
        quality_of_life_scores_data_frame = pd.DataFrame(
            sorted_quality_of_life_scores, index=["Quality of life score"]
        ).transpose()

        # Return the quality of life score data frame
        return quality_of_life_scores_data_frame

    get_quality_of_life_score_for_question_2()
    return (get_quality_of_life_score_for_question_2,)


if __name__ == "__main__":
    app.run()
