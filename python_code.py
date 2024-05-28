# import numpy as np
import pandas as pd

class AutoDataPrep:
    """
    PERFORMS manipulations to a pandas data frame to make ready for data analysis, data science and machine learning.
    """
    def __init__(self):
        """

        Returns
        -------
        None.

        """
        self.df = None
        
    def load(self, file_path, sheet_name=None):
        """
        imports an EXCEL or .CSV file containing the data as a pandas data frame.

        Parameters
        ----------
        file_path : string
            string representing absolute file path or relative file path.
            eg . FilePath = './Filename'
        sheet_name : string
            representing the name of the file, in a multi-sheet file.
            eg. FileName = 'ExcelFileName.xlsx'

        Returns
        -------      
        self.df: Pandas data frame .
        """
        file_ext = (file_path.split("/")[-1]).split(".")[-1]
        if file_ext == "CSV":
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
        elif file_ext == "xlsx":
            # Read all sheets in the excel file
            if sheet_name is not None:
                df = pd.read_excel(file_path,sheet_name)
            else:
                df = pd.read_excel(file_path)
        try:
            self.df = df
            self._properties_ = None
            return df
        except UnboundLocalError:
            print("ERROR IN READING DATA: FORMAT NOT SUPPORTED. FORMAT EXPECTED (.csv & .xlsx)")
            return None

    def properties(self):
        """
        Fetches various parameters about the data such as the data type composition, size of the data,
        number of missing values,
        Parameters
        ----------
        info_type = "Summary": pandas data frame. Outputs a brief summary of the properties of the data.
        Returns
        -------
        """
        df = self.df.copy()
        n_row, n_col = df.shape
        print("\nDATA PROPERTIES : ")
        # print(f"\n{df.head(5)}")
        print(f'\nNUMBER OF DATA ROWS : ({n_row})')
        print(f'NUMBER OF DATA COLUMNS : ({n_col})')
        print('\nDATA COLUMNS : ')
        for col in list(df.columns):
            print(col)
        print('\nAVAILABLE DATA TYPES :')
        # if info_type != "Summary" :
        print(df.dtypes.value_counts())
        print('\nDATA TYPES SUMMARY :')
        for var in df:
            print(f'{var} : {df[var].dtypes}')          

        # INITIALIZE PROPERTIES DICT
        properties_ = {}
        properties = {}

        # CLASSIFY Columns into Categorical & Quantitative Variables.
        cat_data = df.select_dtypes(include="object").copy()
        quant_data = df.select_dtypes(include='number').copy()
        cat_cols = cat_data.columns
        quant_cols = quant_data.columns
        # UPDATE data-class dictionary
        properties_[f"quantifiable ({quant_cols.shape[0]})"] = list(quant_cols)
        properties["quantifiable"] = quant_cols.shape[0]
        # PRINT OUTPUT
        print('\nCATEGORICAL COLUMNS: ')
        for col in list(cat_cols):
            print(col)
        print('\nQUANTITATIVE COLUMNS: ')
        for col in list(quant_cols):
            print(col)
        properties["categorical"] = cat_cols.shape[0]
        properties_[f"categorical ({cat_cols.shape[0]})"] = list(cat_cols)
        # Numerical / Quantitative,  Variables.
        print("\nCOLUMN DATA TYPES: ")
        # print(properties_)
        # print(quant_cols)
        print(list(properties_.keys()))

        # Find the set of columns in the data with missing values.
        df_mean = df.isnull().mean()
        null_cols_names = df.columns[df_mean > 0]
        cols_null_score = df_mean[df_mean > 0]
        print(f'\nNO. OF COLUMNS WITH MISSING VALUES: ({len(null_cols_names)})')
        print('\nCOLUMNS WITH MISSING VALUES:') 
        for col in list(null_cols_names):
            print(col)

        # EVALUATE the number of missing values in each column.
        print('\nNUMBER OF MISSING VALUES PER COLUMN (QTY): ')
        print(df_mean * n_row)
        print('\nNUMBER OF MISSING VALUES PER COLUMN (%):')
        print((df_mean * 100).round(2))
        # self._properties_ =  properties_
          #     print('Broad Data Distribution :')
        # for var in df:
        #     print(df[var].value_counts())
        return cols_null_score * 100

    def prep(self, rename_scheme={}, max_null_threshold_val=25):
        """
        PERFORMS manipulations to a pandas data frame to make ready for data analysis, data science and machine learning.
        ACTIONS:
        - Replaces categorical values with dummy variables.
        - Removes rows with NULL/ NAN values.
        - Drops columns with null values > MAX_NULL_THESHOLD_VAL. eg. 50% of missing values.
        - RENAMES column names with specified column names.
        - TYPE REFORMATTING. Changes data type eg uint8 to float32

        Parameters
        ----------
        df : pandas data frame
             raw pandas data frame, un prepared for modelling

        rename_scheme : Dictionary with variable names as keys and new values as values.
                         rename_scheme is of the form : {Variable Name : [Prev Value, New Value],...}

        Returns
        -------
        df, a new data frame obtained after removing missing rows,
        displaying quantifiable and categorical columns, and general statistics-


        -data about the data frame.
        """
        # initialize our return df variable.
        df = self.df.copy()


        # Get Data Properties.
        null_cols_names = self.properties()

        print("\nDATA PREPARATION : ")

        # Data Imputation
        for var in rename_scheme.keys() :
            # CHECK If data update scheme for that variable is nested.
            if isinstance(rename_scheme[var],dict):
                for prev_val in rename_scheme[var].keys(): 
                    new_val = rename_scheme[var][prev_val]
                    df[var].replace(prev_val, new_val, inplace=True)

            else:
                prev_val, new_val = rename_scheme[var]
                df[var].replace(prev_val, new_val, inplace=True)


        # Type Reformatting
        # df["YearsCodePro"] = df["YearsCodePro"].astype('float64')

        # Remove all rows of columns with NAN values.
        print(
            f'\nDROP COLUMNS with {max_null_threshold_val}% of their values as NAN')
        print(f'\nSHAPE, PRE DATA-PREP.: {df.shape}')
        # DROP COLUMNS containing NAN/ NULL values exceeding max_null_threhold_val
        dropped_cols = []
        for col in null_cols_names.index:
            if null_cols_names[col] > max_null_threshold_val:
                dropped_cols.append(col)
                df = df.drop(col, axis=1)
        print('\nCOLUMNS DROPPED: ')
        for col in dropped_cols:
            print(col)
        print(f'\nSHAPE, POST DATA-PREP: {df.shape}')
        #DROP ROWS containing NAN values')
        df = df.dropna(axis=0)
        print(f'\nSHAPE, POST NAN ROWS REMOVAL: {df.shape}')
  
        # REPLACE Categorical Variables with Dummy Variables.
        cat_vars = df.select_dtypes(include=['object']).copy()
        cat_cols = cat_vars.columns
        for var in cat_cols:
            dummy_var = pd.get_dummies(
                df[var], prefix=var, prefix_sep='_', drop_first=True)
            df = pd.concat(
                [df.drop(var, axis=1), dummy_var], axis=1)
        print(f'\nSHAPE, POST SUBSTITUTION of categorical variables with dummy variables : {df.shape}')
        
        return df
