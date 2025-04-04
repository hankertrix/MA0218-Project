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
app = marimo.App(width="medium")


@app.cell
def _():
    # Import all the required libraries
    import re

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sb
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
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
    from sklearn.svm import SVR, LinearSVR
    from sklearn.tree import DecisionTreeRegressor

    # Set the seaborn style
    sb.set_theme()
    return (
        BayesianRidge,
        DecisionTreeRegressor,
        ElasticNet,
        ExtraTreesRegressor,
        HuberRegressor,
        IterativeImputer,
        KNeighborsRegressor,
        Lasso,
        LinearRegression,
        LinearSVR,
        PassiveAggressiveRegressor,
        RANSACRegressor,
        RandomForestRegressor,
        Ridge,
        SGDRegressor,
        SVR,
        TheilSenRegressor,
        enable_iterative_imputer,
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
def _(np):
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
    YEAR_RANGE = np.array(list(range(1990, 2010 + 1)))
    YEAR_RANGE_STR = [str(year) for year in YEAR_RANGE]
    return (
        COLUMNS_TO_DROP,
        DATA_FILE,
        REGIONS_FOR_PROBLEM_1,
        REGIONS_TO_REMOVE_FOR_PROBLEM_2,
        SERIES_CODES_PROBLEM_1,
        SERIES_CODES_PROBLEM_2,
        YEAR_RANGE,
        YEAR_RANGE_STR,
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
    IterativeImputer,
    LinearRegression,
    YEAR_RANGE_STR,
    html,
    pd,
    strip_unnecessary_code,
):
    # Create the function to impute the missing data
    def impute_missing_data(given_data: pd.DataFrame) -> pd.DataFrame:
        "Function to impute the missing data row by row."

        # Initialise the imputer object
        imputer_object = IterativeImputer(estimator=LinearRegression())

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
    BayesianRidge,
    DecisionTreeRegressor,
    ElasticNet,
    ExtraTreesRegressor,
    HuberRegressor,
    KNeighborsRegressor,
    Lasso,
    LinearRegression,
    LinearSVR,
    RANSACRegressor,
    RandomForestRegressor,
    Ridge,
    SGDRegressor,
    SVR,
    YEAR_RANGE,
    YEAR_RANGE_STR,
    cleaned_data,
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

        # The training data for the series
        y_train = data[YEAR_RANGE_STR].values.flatten()

        # Regression models
        regression_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
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
            "SGD Regression": SGDRegressor(),
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
        }

        # Get the number of plots on each side
        number_of_plots = int(np.ceil(np.sqrt(len(regression_models))))

        # Create the figure and the subplots
        figure, axes = plt.subplots(
            number_of_plots, number_of_plots, figsize=(32, 32)
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

        # Put the data into a data frame
        results_dataframe = (
            pd.DataFrame(results)
            .sort_values(by="R squared score", ascending=False)
            .reset_index(drop=True)
        )

        # Return the results dataframe
        return results_dataframe, figure

    # Run the exploratory analysis
    exploratory_analysis()
    return (exploratory_analysis,)


@app.cell
def _(mo):
    mo.md(r"### Are we getting greener or more sustainable?")
    return


@app.cell
def _(problem_1_data):
    problem_1_data
    return


@app.cell
def _(model, plt, timeline, world_data):
    def _():
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
        return plt.show()

    _()
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
def _(model, plt, problem_1_data, timeline):
    def _():
        for j in problem_1_data:
            f, axes = plt.subplots(1, 8, figsize=(36, 8))
            jdata = problem_1_data[j]
            counti = 0
            print("Country: ", j)
            for i in jdata:
                datai = jdata[i]
                print(datai.describe())
                # train linreg
                model.fit(timeline, datai)
                regline_x = timeline
                regline_y = model.predict(regline_x)

                # visualise the data regression
                axes[counti].scatter(timeline, datai)
                axes[counti].plot(regline_x, regline_y, "r-", linewidth=3)
                axes[counti].set_xlabel("Timeline")
                axes[counti].set_ylabel(i)
                counti += 1
        return plt.show()

    _()
    return


@app.cell
def _(DecisionTreeRegressor, LinearSVR, Ridge, np, pd, plt, problem_2_data):
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
    return (
        co2_data,
        decision_tree_model,
        linear_model,
        new_years,
        prediction,
        svr_model,
        train_data,
        years,
        years_str,
    )


@app.cell
def _(mo):
    mo.md(r"### Which should i move to in the future?")
    return


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
def _(LinearRegression, np, pd, problem_2_data, timeline):
    # Extracting exact sets of data from problem_2_data for analysis
    predictions_df = pd.DataFrame()
    row = 0
    column = 0
    for country in problem_2_data:
        # Extract the data for the country
        Country_data = pd.DataFrame(problem_2_data[str(country)])
        row += 1
        country_predictions = {}  # Temporary dictionary to store predictions for this country
        for series in Country_data:
            Series_data = pd.DataFrame(Country_data[str(series)])
            linreg = LinearRegression()
            linreg.fit(timeline, Series_data)
            predicted_value_2025 = linreg.predict(np.array([[2025]]))
            country_predictions[series] = predicted_value_2025[0][
                0
            ]  # Store the prediction
            column += 1
        predictions_df = predictions_df._append(
            pd.Series(country_predictions, name=country)
        )

    # Print the final DataFrame
    predictions_df

    return (
        Country_data,
        Series_data,
        column,
        country,
        country_predictions,
        linreg,
        predicted_value_2025,
        predictions_df,
        row,
        series,
    )


if __name__ == "__main__":
    app.run()
