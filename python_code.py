from cgi import print_form
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import AllTogether as t
import seaborn as sns


class AutoDataIRP:
    def __init__():
        """
        

        Returns
        -------
        None.

        """

        
    def load(self, file_path, sheet_name=""):
        """
        Loads data file and returns a pandas data frame

        Parameters
        ----------
        file_path : string
            string representing absolute file path or relative file path.
            eg . FilePath = 'C:/Your/File/Path'
        sheet_name : string
            representing the name of the file, in a multi-sheet file.
            eg. FileName = 'ExcelFileName.xlsx'

        Returns
        -------      
        self.df: Pandas data frame .
        """
        # Read all sheets in the excel file
        df = pd.read_excel(file_path,sheet_name="IPLSOA")
        # df = pd.read_csv(dir)...

        self.df = df
        return df

    def stats(self, info_type="Summary"):
        """
        Parameters
        ----------
        df : pandas data frame.

        Returns
        -------
        """
        print("\nData Stats : ")
        df = self.df
        try:
            n_row, n_col = df.shape

        except ValueError:
            n_row = 'generic'
            n_col = 'generic'


        if info_type == "Full" :
            print_form()


            num_rows = "{:0,.0f}".format(n_row)
            print(f'Data rows - ({num_rows}), and Data columns - ({n_col}).\n')
            print('Available Data Types :')
            print_form()
            print(df.dtypes.value_counts())

            print_form()
            print('Broad Data Distribution :')
            for var in df:
                print_form()
                print(df[var].value_counts())


        # Classifying Columns into Categorical & Quantitative Variables.
        data_class_summary = {}
        data_class = {}
        # Categorical Variables.
        cat_data = df.select_dtypes(include="object").copy()
        cat_cols = cat_data.columns
        data_class["categorical"] = cat_cols.shape[0]
        data_class_summary[f"categorical ({cat_cols.shape[0]})"] = list(cat_cols)
        # Numerical / Quantitative,  Variables.
        num_data = df.select_dtypes(include='number').copy()
        num_cols = num_data.columns
        data_class_summary[f"quantifiable ({num_cols.shape[0]})"] = list(num_cols)
        data_class["quantifiable"] = num_cols.shape[0]
        print("\nData Class Summary : ")
        
        # print(data_class_summary)
        # print(num_cols)
        print(list(data_class_summary.keys()))

        # Find the set of columns in the data with missing values.
        df_mean = df.isnull().mean()
        null_cols = set(df.columns[df_mean > 0])
        print(f"\nColumns with missing values: ({len(null_cols)})")
        print(f"\nColumns with missing values: ({null_cols})")
        

        # Evaluate the number of missing values in each column.
        null_cols_stats = df_mean * n_row
        notnull_cols = null_cols_stats[null_cols_stats > 0]
        print('\nNumber of missing values per Column: ')
        
        if notnull_cols.empty:
            print(null_cols_stats)
        else:
            print(notnull_cols)
        print('\nNumber of missing values per Column (%):')
        
        missing_values_ratio = (df_mean[df_mean > 0] * 100).round(2)
        if missing_values_ratio.empty:
            print("None")
        else:
            print(missing_values_ratio)

        return missing_values_ratio


        
    
    def prep(self,variables_dict):
        """
        Parameters
        ----------
        Takes a pandas data frame and replaces values under the given variable name columns
        with new values
        df : pandas data frame
             raw pandas data frame, un prepared for modelling

        variables_dict : Dictionary with variable names as keys and new values as values.
                         variables_dict is of the form : {Variable Name : [Prev Value, New Value],...}

        Returns
        -------
        prepared_df, a new data frame obtained after removing missing rows,
        displaying quantifiable and categorical columns, and general statistics-


        -data about the data frame.
        """
        # initialize our return df variable.
        prepared_df = self.df.copy()


        # Get data stats.
        missing_values_ratio = self.data_stats(prepared_df, info_type="Full")

        # Data Imputation
        for var in variables_dict.keys() :
            # If data update for that variable is nested.
            if type(variables_dict[var])== dict:
                for prev_var_val in variables_dict[var].keys(): 
                    new_var_val = variables_dict[var][prev_var_val]
                    prepared_df[var].replace(prev_var_val, new_var_val, inplace=True)

            else:
                prev_var_val, new_var_val = variables_dict[var]
                prepared_df[var].replace(prev_var_val, new_var_val, inplace=True)


        # Type Reformatting
        # prepared_df["YearsCodePro"] = prepared_df["YearsCodePro"].astype('float64')

        # Remove all rows of columns with NAN values.
        threshold = 60
        print(
            f'\nData-Prep: Drop columns with {threshold}% of their values as NAN')
        print(f'\nShape, pre data-prep: {prepared_df.shape}')
        for col in missing_values_ratio.index:
            if missing_values_ratio[col] > threshold:
                prepared_df = prepared_df.drop(col, axis=1)
        print(f'Shape, post data-prep: {prepared_df.shape}')
        print('\nData-Prep: Drop rows containing NAN values')
        prepared_df = prepared_df.dropna(axis=0)
        print(f'Shape, Post removal of NAN column rows: {prepared_df.shape}')

        # Replacing Categorical Variables with Dummy Variables.
        print('\nData-Prep: Replace Categorical Variables with Dummy Variables')
        cat_vars = prepared_df.select_dtypes(include=['object']).copy()
        cat_cols = cat_vars.columns
        for var in cat_cols:
            dummy_var = pd.get_dummies(
                prepared_df[var], prefix=var, prefix_sep='_', drop_first=True)
            prepared_df = pd.concat(
                [prepared_df.drop(var, axis=1), dummy_var], axis=1)
        print(f'Shape, Post replacement of Categorical Variables with Dummy Variables : {prepared_df.shape}')
        self.data_stats(prepared_df)

        return prepared_df
        
    def aprintend():
        """
        """
        
    def data_plot(x_label, y_label, x_values, y_values, path):
        """


        Parameters
        ----------
        x_label : TYPE
            DESCRIPTION.
        y_label : TYPE
            DESCRIPTION.
        x_values : TYPE
            DESCRIPTION.
        y_values : TYPE
            DESCRIPTION.
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        print("\n\nData Plots")
        print_form()

        if type(x_values[0]) == list or type(x_values[0]) == pd.core.series.Series:
            colors = ['b', 'g', 'c', 'm', 'y', 'k']
            fig, ax = plt.subplots()
            for index in range(0, len(x_values)):
                x_val = x_values[index]
                y_val = y_values[index]
                LABEL = 'Model Prediction'

                if index == 0:
                    # Add regression line to plot
                    best_fit_params = np.polyfit(x_val, y_val, 1)
                    best_fit_poly = np.poly1d(best_fit_params)
                    ax.plot(x_val, best_fit_poly(x_val),
                            color='r', label='Line Of Best Fit')
                    LABEL = 'Actual'

                ax.set_xlabel(f'{x_label}')
                ax.set_ylabel(f'{y_label}')
                ax.set_title(f'{x_label} vs {y_label}')

                color = colors[min(index, len(colors))]
                ax.plot(x_val, y_val, 'o-', color=color, label=LABEL)
                ax.legend()


        else:
            # Add regression line to plot
            best_fit_params = np.polyfit(x_values, y_values, 1)
            best_fit_poly = np.poly1d(best_fit_params)

            plt.xlabel(f'{x_label}')
            plt.ylabel(f'{y_label}')
            plt.title(f'{x_label} vs {y_label}')

            plt.plot(x_values, y_values, 'o-', color='b')
            plt.plot(x_values, y_values, 'o-', color='b')
            plt.plot(x_values, best_fit_poly(x_values), color='red')
        plt.savefig(path)
        plt.show()

        ƒƒƒ