{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='pink'>**Summative Assignment - Nicole Ben Haim & Noam Ifargan**</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "import re\n",
    "import pandas as pd\n",
    "import numpy as np \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "datafile = 'C://Users/noami/Desktop/Summative Assignment/output_all_students_Train_v10.xlsx'\n",
    "data1 = pd.read_excel(datafile)\n",
    "data = data1.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='red'>**STEP 1 - data cleaning and preparation**</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Delete properties with missing prices - \n",
    "Remove properties from the dataset where the price is missing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>City</th>\n",
       "      <th>type</th>\n",
       "      <th>room_number</th>\n",
       "      <th>Area</th>\n",
       "      <th>Street</th>\n",
       "      <th>number_in_street</th>\n",
       "      <th>city_area</th>\n",
       "      <th>price</th>\n",
       "      <th>num_of_images</th>\n",
       "      <th>floor_out_of</th>\n",
       "      <th>...</th>\n",
       "      <th>hasStorage</th>\n",
       "      <th>condition</th>\n",
       "      <th>hasAirCondition</th>\n",
       "      <th>hasBalcony</th>\n",
       "      <th>hasMamad</th>\n",
       "      <th>handicapFriendly</th>\n",
       "      <th>entranceDate</th>\n",
       "      <th>furniture</th>\n",
       "      <th>publishedDays</th>\n",
       "      <th>description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>5.5</td>\n",
       "      <td>137</td>\n",
       "      <td>רפאלי שרגא</td>\n",
       "      <td>3</td>\n",
       "      <td>אם המושבות החדשה</td>\n",
       "      <td>3600000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 11 מתוך 19</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>שמור</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>7</td>\n",
       "      <td>למכירה 5.5 חדרים ענקית, מרווחת , מוארת , קומה ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>3</td>\n",
       "      <td>84</td>\n",
       "      <td>כצנלסון אהרון</td>\n",
       "      <td>6</td>\n",
       "      <td>נווה גן</td>\n",
       "      <td>2550000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 6 מתוך 9</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>שמור</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>מיידי</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>8</td>\n",
       "      <td>למכירה מפרטי ברחוב אהרון כצנלסון השקט והמבוקש,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4</td>\n",
       "      <td>120</td>\n",
       "      <td>הירקונים</td>\n",
       "      <td>17</td>\n",
       "      <td>קרית הרב סלומון</td>\n",
       "      <td>2650000</td>\n",
       "      <td>10.0</td>\n",
       "      <td>קומה 2 מתוך 7</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>חדש</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>חלקי</td>\n",
       "      <td>6</td>\n",
       "      <td>פריים לוקשיין בשכונת שיפר המאוד מבוקשת!!! למכי...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>3.5</td>\n",
       "      <td>110</td>\n",
       "      <td>סלנט שמואל</td>\n",
       "      <td>56</td>\n",
       "      <td>המרכז השקט</td>\n",
       "      <td>2450000</td>\n",
       "      <td>8.0</td>\n",
       "      <td>קומה 2 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>משופץ</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>חלקי</td>\n",
       "      <td>8</td>\n",
       "      <td>בפתח תקווה ברחוב שמואל סלנט המבוקש,לא כדאי לפס...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4.5</td>\n",
       "      <td>120</td>\n",
       "      <td>בן צבי יצחק</td>\n",
       "      <td>28</td>\n",
       "      <td>כפר גנים ב</td>\n",
       "      <td>2720000</td>\n",
       "      <td>9.0</td>\n",
       "      <td>קומה 3 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>False</td>\n",
       "      <td>משופץ</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>21</td>\n",
       "      <td>הדירה משופצת חלקית בטעם טוב , פונה לעורף לכיוו...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>694</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>בית פרטי</td>\n",
       "      <td>9.5 חד׳</td>\n",
       "      <td>350 מ\"ר</td>\n",
       "      <td>הורד</td>\n",
       "      <td>35</td>\n",
       "      <td>2005</td>\n",
       "      <td>8200000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קומה 4 מתוך 4</td>\n",
       "      <td>...</td>\n",
       "      <td>יש מחסן</td>\n",
       "      <td>שמור</td>\n",
       "      <td>יש מיזוג אויר</td>\n",
       "      <td>יש מרפסת</td>\n",
       "      <td>יש ממ״ד</td>\n",
       "      <td>לא נגיש לנכים</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>אין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>בית גדול מאוד , 3 כיווני אוויר , 2 מרפסות גדולות</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>695</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4 חד׳</td>\n",
       "      <td>110 מ\"ר</td>\n",
       "      <td>קזן</td>\n",
       "      <td>NaN</td>\n",
       "      <td>מרכז דרום</td>\n",
       "      <td>3350000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 4 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>יש מחסן</td>\n",
       "      <td>חדש</td>\n",
       "      <td>יש מיזוג אויר</td>\n",
       "      <td>יש מרפסת</td>\n",
       "      <td>יש ממ״ד</td>\n",
       "      <td>נגיש לנכים</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>בקומה 4 הדירה 110 מ\"ר נטו136 מ\"ר ארנונה.מעוצבת...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>696</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>קוטג'</td>\n",
       "      <td>7 חד׳</td>\n",
       "      <td>376 מ\"ר</td>\n",
       "      <td>הטללים</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קרית גנים</td>\n",
       "      <td>8500000</td>\n",
       "      <td>13.0</td>\n",
       "      <td>קומת קרקע</td>\n",
       "      <td>...</td>\n",
       "      <td>אין מחסן</td>\n",
       "      <td>חדש</td>\n",
       "      <td>אין מיזוג אויר</td>\n",
       "      <td>אין מרפסת</td>\n",
       "      <td>אין ממ״ד</td>\n",
       "      <td>לא נגיש לנכים</td>\n",
       "      <td>2023-08-01 00:00:00</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>במערב המבוקש! !ה---- בית הנדיר הזה הוא עוד אחד...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>697</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>5 חד׳</td>\n",
       "      <td>126 מ\"ר</td>\n",
       "      <td>אחד העם</td>\n",
       "      <td>NaN</td>\n",
       "      <td>לסטר</td>\n",
       "      <td>3850000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קומה 5 מתוך 7</td>\n",
       "      <td>...</td>\n",
       "      <td>אין מחסן</td>\n",
       "      <td>חדש</td>\n",
       "      <td>יש מיזוג אויר</td>\n",
       "      <td>יש מרפסת</td>\n",
       "      <td>יש ממ״ד</td>\n",
       "      <td>נגיש לנכים</td>\n",
       "      <td>מיידי</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>דירה חדשה מהקבלן באזור מבוקש .\\n5 חדרים .מרווח...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>698</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4.5 חד׳</td>\n",
       "      <td>140 מ\"ר</td>\n",
       "      <td>קזן</td>\n",
       "      <td>10</td>\n",
       "      <td>מרכז דרום</td>\n",
       "      <td>3730000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 3 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>יש מחסן</td>\n",
       "      <td>משופץ</td>\n",
       "      <td>יש מיזוג אויר</td>\n",
       "      <td>יש מרפסת</td>\n",
       "      <td>יש ממ״ד</td>\n",
       "      <td>נגיש לנכים</td>\n",
       "      <td>2024-02-01 00:00:00</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>לא למתווכים!, ללא תיווך!, מתווכים לא להתקשר! ד...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>694 rows × 23 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          City      type room_number     Area         Street number_in_street  \\\n",
       "0    פתח תקווה      דירה         5.5      137     רפאלי שרגא                3   \n",
       "1    פתח תקווה      דירה           3       84  כצנלסון אהרון                6   \n",
       "2    פתח תקווה      דירה           4      120       הירקונים               17   \n",
       "3    פתח תקווה      דירה         3.5      110     סלנט שמואל               56   \n",
       "4    פתח תקווה      דירה         4.5      120    בן צבי יצחק               28   \n",
       "..         ...       ...         ...      ...            ...              ...   \n",
       "694      רעננה  בית פרטי     9.5 חד׳  350 מ\"ר           הורד               35   \n",
       "695      רעננה      דירה       4 חד׳  110 מ\"ר            קזן              NaN   \n",
       "696      רעננה     קוטג'       7 חד׳  376 מ\"ר         הטללים              NaN   \n",
       "697      רעננה      דירה       5 חד׳  126 מ\"ר        אחד העם              NaN   \n",
       "698      רעננה      דירה     4.5 חד׳  140 מ\"ר            קזן               10   \n",
       "\n",
       "            city_area    price  num_of_images     floor_out_of  ...  \\\n",
       "0    אם המושבות החדשה  3600000            6.0  קומה 11 מתוך 19  ...   \n",
       "1             נווה גן  2550000            6.0    קומה 6 מתוך 9  ...   \n",
       "2     קרית הרב סלומון  2650000           10.0    קומה 2 מתוך 7  ...   \n",
       "3          המרכז השקט  2450000            8.0    קומה 2 מתוך 6  ...   \n",
       "4          כפר גנים ב  2720000            9.0    קומה 3 מתוך 6  ...   \n",
       "..                ...      ...            ...              ...  ...   \n",
       "694              2005  8200000            NaN    קומה 4 מתוך 4  ...   \n",
       "695         מרכז דרום  3350000            6.0    קומה 4 מתוך 6  ...   \n",
       "696         קרית גנים  8500000           13.0        קומת קרקע  ...   \n",
       "697              לסטר  3850000            NaN    קומה 5 מתוך 7  ...   \n",
       "698         מרכז דרום  3730000            6.0    קומה 3 מתוך 6  ...   \n",
       "\n",
       "    hasStorage  condition  hasAirCondition  hasBalcony  hasMamad   \\\n",
       "0         False       שמור             True       False      True   \n",
       "1          True       שמור             True       False      True   \n",
       "2          True        חדש             True        True      True   \n",
       "3         False      משופץ             True       False      True   \n",
       "4         False      משופץ             True        True      True   \n",
       "..          ...        ...              ...         ...       ...   \n",
       "694     יש מחסן       שמור    יש מיזוג אויר    יש מרפסת   יש ממ״ד   \n",
       "695     יש מחסן        חדש    יש מיזוג אויר    יש מרפסת   יש ממ״ד   \n",
       "696    אין מחסן        חדש   אין מיזוג אויר   אין מרפסת  אין ממ״ד   \n",
       "697    אין מחסן        חדש    יש מיזוג אויר    יש מרפסת   יש ממ״ד   \n",
       "698     יש מחסן      משופץ    יש מיזוג אויר    יש מרפסת   יש ממ״ד   \n",
       "\n",
       "    handicapFriendly         entranceDate  furniture  publishedDays   \\\n",
       "0                True                 גמיש   לא צויין              7   \n",
       "1               False                מיידי   לא צויין              8   \n",
       "2               False                 גמיש       חלקי              6   \n",
       "3                True                 גמיש       חלקי              8   \n",
       "4                True                 גמיש   לא צויין             21   \n",
       "..                ...                  ...        ...            ...   \n",
       "694     לא נגיש לנכים                 גמיש        אין            NaN   \n",
       "695        נגיש לנכים                 גמיש   לא צויין            NaN   \n",
       "696     לא נגיש לנכים  2023-08-01 00:00:00   לא צויין            NaN   \n",
       "697        נגיש לנכים                מיידי   לא צויין            NaN   \n",
       "698        נגיש לנכים  2024-02-01 00:00:00   לא צויין            NaN   \n",
       "\n",
       "                                          description   \n",
       "0    למכירה 5.5 חדרים ענקית, מרווחת , מוארת , קומה ...  \n",
       "1    למכירה מפרטי ברחוב אהרון כצנלסון השקט והמבוקש,...  \n",
       "2    פריים לוקשיין בשכונת שיפר המאוד מבוקשת!!! למכי...  \n",
       "3    בפתח תקווה ברחוב שמואל סלנט המבוקש,לא כדאי לפס...  \n",
       "4    הדירה משופצת חלקית בטעם טוב , פונה לעורף לכיוו...  \n",
       "..                                                 ...  \n",
       "694   בית גדול מאוד , 3 כיווני אוויר , 2 מרפסות גדולות  \n",
       "695  בקומה 4 הדירה 110 מ\"ר נטו136 מ\"ר ארנונה.מעוצבת...  \n",
       "696  במערב המבוקש! !ה---- בית הנדיר הזה הוא עוד אחד...  \n",
       "697  דירה חדשה מהקבלן באזור מבוקש .\\n5 חדרים .מרווח...  \n",
       "698  לא למתווכים!, ללא תיווך!, מתווכים לא להתקשר! ד...  \n",
       "\n",
       "[694 rows x 23 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = data.dropna(subset=['price'])\n",
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2+3. Ensure that the price  and area is numeric:\n",
    "Keep only the numeric values for the price column and Convert the price values to a numeric data type.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-4-2f7f79d3a24d>:17: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['Area'] = data['Area'].apply(clean_price_and_Area).replace('', np.nan)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['137', '84', '120', '110', '98', '140', '147', '100', '75', '95',\n",
       "       '125', '121', '55', '70', '145', '62', '93', '108', '49', '150',\n",
       "       '130', '104', nan, '78', '270', '90', '96', '135', '80', '118',\n",
       "       '119', '148', '52', '54', '182', '85', '300', '53', '122', '76',\n",
       "       '160', '123', '144', '83', '73', '127', '151', '184', '134', '152',\n",
       "       '74', '64', '67', '65', '138', '46', '139', '79', '92', '68',\n",
       "       '146', '103', '106', '132', '105', '117', '175', '99', '225',\n",
       "       '113', '112', '60', '220', '126', '128', '114', '167', '94', '355',\n",
       "       '172', '235', '174', '143', '200', '280', '250', '136', '109',\n",
       "       '330', '180', '45', '240', '40', '89', '157', '173', '170', '176',\n",
       "       '77', '50', '141', '124', '286', '81', '1000', '318', '86', '102',\n",
       "       '116', '155', '131', '42', '111', '165', '56', '339', '504', '101',\n",
       "       '91', '115', '57', '71', '319', '87', '30', '69', '97', '208',\n",
       "       '350', '234', '161', '260', '82', '142', '129', '156', '107', '88',\n",
       "       '315', '380', '158', '72', '154', '48', '159', '63', '58', '149',\n",
       "       '153', '376'], dtype=object)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def clean_price_and_Area(x):\n",
    "    if isinstance(x, str):\n",
    "        x = re.sub(r'[^\\d,]', '', x)  # מחיקת תווים שאינם ספרות או פסיק\n",
    "        return x.replace(',', '')  # הסרת פסיקים\n",
    "    elif isinstance(x, int):\n",
    "        return str(x)  # חיזור ערך מספרי כמחרוזת\n",
    "    else:\n",
    "        return ''\n",
    "    \n",
    "#ניקוי עמודת המחיר   \n",
    "data['price'] = data['price'].apply(clean_price_and_Area).replace('', np.nan)\n",
    "data = data.dropna(subset=['price'])\n",
    "unique_price = data['price'].unique()\n",
    "unique_price\n",
    "\n",
    "#ניקוי עמודת השטח\n",
    "data['Area'] = data['Area'].apply(clean_price_and_Area).replace('', np.nan)\n",
    "unique_Area = data['Area'].unique()\n",
    "unique_Area"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.series.Series'>\n",
      "Int64Index: 693 entries, 0 to 698\n",
      "Series name: price\n",
      "Non-Null Count  Dtype\n",
      "--------------  -----\n",
      "693 non-null    int32\n",
      "dtypes: int32(1)\n",
      "memory usage: 8.1 KB\n",
      "<class 'pandas.core.series.Series'>\n",
      "Int64Index: 693 entries, 0 to 698\n",
      "Series name: Area\n",
      "Non-Null Count  Dtype\n",
      "--------------  -----\n",
      "677 non-null    Int64\n",
      "dtypes: Int64(1)\n",
      "memory usage: 11.5 KB\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-5-d8bf79847379>:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['price'] = data['price'].astype(int)\n",
      "<ipython-input-5-d8bf79847379>:4: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['Area'] = pd.to_numeric(data['Area'], errors='coerce').fillna(data['Area'])\n",
      "<ipython-input-5-d8bf79847379>:5: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['Area'] = pd.to_numeric(data['Area'], errors='coerce').astype('Int64')\n"
     ]
    }
   ],
   "source": [
    "#המרת השדות לערך מספרי\n",
    "data['price'] = data['price'].astype(int)\n",
    "data['price'].info()\n",
    "data['Area'] = pd.to_numeric(data['Area'], errors='coerce').fillna(data['Area'])\n",
    "data['Area'] = pd.to_numeric(data['Area'], errors='coerce').astype('Int64')\n",
    "data['Area'].info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4. Remove unnecessary punctuation from text fields:\n",
    "\n",
    "Remove excessive commas or other unnecessary punctuation marks from address, region, property description, etc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-6-950d2137068a>:11: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['Street']=data['Street'].apply(lambda x: word(x))\n",
      "<ipython-input-6-950d2137068a>:12: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['city_area']=data['city_area'].apply(lambda x: word(x))\n",
      "<ipython-input-6-950d2137068a>:13: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['description ']=data['description '].apply(lambda x: word(x))\n"
     ]
    }
   ],
   "source": [
    "def word(x):\n",
    "    if isinstance(x, str):\n",
    "        x = re.sub(r'[^\\w /.]', '', x)  # מחיקת תווים שאינם ספרות או פסיק\n",
    "        x=x.replace('/',' ')\n",
    "        x=x.replace('.',' ')\n",
    "        x=x.replace('  ',' ')\n",
    "        return x.strip()\n",
    "    else:\n",
    "        return ''\n",
    "\n",
    "data['Street']=data['Street'].apply(lambda x: word(x))\n",
    "data['city_area']=data['city_area'].apply(lambda x: word(x))\n",
    "data['description ']=data['description '].apply(lambda x: word(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "5. Add a floor column:\n",
    "\n",
    "Extract the floor information from the \"floor_out_of\" column and add it as a separate column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-7-afcf27eef58a>:13: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['floor']=data['floor_out_of'].apply(lambda x:get_floor(x) ).astype('int64')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([11,  6,  2,  3,  7,  4, 13,  9, 14,  0,  1,  5, 10,  8, 19, 15, 25,\n",
       "       17, 12, 18, 29, 16, 20], dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def get_floor(x):\n",
    "    if isinstance(x, str):         \n",
    "        x_list=x.split()\n",
    "        if len(x_list)>1:\n",
    "            if x_list[1]=='קרקע' or x_list[1]=='מרתף': \n",
    "                return 0       # 'קומת קרקע/מרתף', 'קומת קרקע מתוך 5'\n",
    "            else:\n",
    "                return x_list[1]    # 'קומה 2 מתוך 4', 'קומה 6'\n",
    "        else:\n",
    "            return 0   # הגיוני שלא מילאו בגלל שהבית לא נמצא בבניין. לכן נמלא 0 \n",
    "    else:\n",
    "        return 0  # for nan # הגיוני שלא מילאו בגלל שהבית לא נמצא בבניין. לכן נמלא 0 \n",
    "data['floor']=data['floor_out_of'].apply(lambda x:get_floor(x) ).astype('int64')\n",
    "data['floor'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "6. Add a total_floors column:\n",
    "\n",
    "Calculate the total number of floors in the building and add it as a separate column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-8-ec73e450e95d>:16: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['total_floors']=data['floor_out_of'].apply(lambda x: get_totalfloor(x)).astype('int64')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([19,  9,  7,  6,  8, 24, 14, 16, 21, 28,  3,  0,  4,  2, 10,  5, 18,\n",
       "       20, 13,  1, 12, 30, 15, 26, 25, 17, 11, 27, 22], dtype=int64)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def get_totalfloor(x):\n",
    "    if isinstance(x, str):         \n",
    "        x_list=x.split()\n",
    "        if len(x_list)==2:\n",
    "            if x_list[1]=='קרקע' or x_list[1]=='מרתף':   \n",
    "                return 0  # 'קומת קרקע'\n",
    "            else:\n",
    "                return x_list[1]   # 'קומה 6'\n",
    "        if len(x_list)==4:\n",
    "            return x_list[3]    # 'קומה 4 מתוך 6', 'קומת קרקע מתוך 5'\n",
    "            \n",
    "        else:\n",
    "            return 0   # הגיוני שלא מילאו בגלל שהבית לא נמצא בבניין. לכן נמלא 0 \n",
    "    else:\n",
    "        return 0  # for nan\n",
    "data['total_floors']=data['floor_out_of'].apply(lambda x: get_totalfloor(x)).astype('int64')\n",
    "data['total_floors'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "7. Create an \"entrance_date\" categorical column:\n",
    "\n",
    "Based on the current date, categorize the entrance date into one of the following categories:\n",
    "Less than 6 months\n",
    "6-12 months\n",
    "Above 1 year\n",
    "Flexible\n",
    "Not defined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-9-aa2536bd2e8d>:24: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data['entrance_date']=data['entranceDate '].apply(lambda x: entertime(x))\n"
     ]
    }
   ],
   "source": [
    "def entertime(x):\n",
    "    if isinstance(x, str):\n",
    "        x.strip()\n",
    "        if x=='גמיש':\n",
    "            return 'flexible'\n",
    "        if x=='לא צויין':\n",
    "            return 'not_defined'\n",
    "        if x=='מיידי':\n",
    "            return 'less_than_6_months'\n",
    "    \n",
    "    if isinstance(x, datetime.datetime):\n",
    "        today = datetime.date.today()\n",
    "        months_more_6 = today + datetime.timedelta(days=6*30)\n",
    "        months_more_12 = today + datetime.timedelta(days=12*30)\n",
    "        if x.date() < months_more_6:\n",
    "            return 'less_than_6_months'\n",
    "        if months_more_6 <= x.date() <= months_more_12:\n",
    "            return 'months_6_12'\n",
    "        if x.date() > months_more_12:\n",
    "            return 'above_year'    \n",
    "    \n",
    "    else:\n",
    "        return 'not_defined'\n",
    "data['entrance_date']=data['entranceDate '].apply(lambda x: entertime(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "8. Represent boolean fields (hasBalcony, hasMamad and other) as 0s and 1s:\n",
    "\n",
    "Convert all boolean fields to 0s and 1s, where 0 represents False and 1 represents True."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# הכנה ל 0 ו 1\n",
    "\n",
    "mappingdata = {\n",
    "    True: 1,\n",
    "    False: 0,\n",
    "    'לא': 0,\n",
    "    'כן': 1,\n",
    "    'yes': 1,\n",
    "    'יש': 1,\n",
    "    'אין': 0,\n",
    "    'no': 0,\n",
    "}\n",
    "\n",
    "data = data.apply(lambda x: x.replace(mappingdata))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# טיפול ממ\"ד 0,1\n",
    "\n",
    "mappingMamad = {\n",
    "    'אין ממ״ד': 0,\n",
    "    'יש ממ״ד': 1,\n",
    "    'יש ממ\"ד': 1,\n",
    "    'אין ממ\"ד': 0\n",
    "}\n",
    "\n",
    "data['hasMamad '] = data['hasMamad '].replace(mappingMamad)\n",
    "data['hasMamad '] = data['hasMamad '].fillna(0)\n",
    "data['hasMamad '] = data['hasMamad '].astype(int)\n",
    "data['hasMamad '].unique()\n",
    "\n",
    "# טיפול במרפסת 0,1\n",
    "\n",
    "mappingBalcony = {\n",
    "    'אין מרפסת': 0,\n",
    "    'יש מרפסת': 1\n",
    "\n",
    "}\n",
    "\n",
    "data['hasBalcony '] = data['hasBalcony '].replace(mappingBalcony)\n",
    "data['hasBalcony '] = data['hasBalcony '].fillna(0)\n",
    "data['hasBalcony '].unique()\n",
    "# טיפול במיזוג אוויר 0,1\n",
    "\n",
    "mappingAC = {\n",
    "    'אין מיזוג אויר': 0,\n",
    "    'יש מיזוג אויר': 1,\n",
    "    'יש מיזוג אוויר': 1\n",
    "}\n",
    "\n",
    "data['hasAirCondition '] = data['hasAirCondition '].replace(mappingAC)\n",
    "data['hasAirCondition '].unique()\n",
    "# טיפול במעלית 0,1\n",
    "\n",
    "mappingElevator = {\n",
    "    'אין מעלית': 0,\n",
    "    'יש מעלית': 1}\n",
    "\n",
    "data['hasElevator '] = data['hasElevator '].replace(mappingElevator)\n",
    "data['hasElevator '].unique()\n",
    "\n",
    "# טיפול חניה 0,1\n",
    "\n",
    "mappingParking = {\n",
    "    'יש חניה': 1,\n",
    "    'יש חנייה': 1,\n",
    "    'אין חניה': 0}\n",
    "\n",
    "data['hasParking '] = data['hasParking '].replace(mappingParking)\n",
    "data['hasParking '].unique()\n",
    "\n",
    "mappingBars = {\n",
    "    'אין סורגים': 0,\n",
    "    'יש סורגים': 1\n",
    "}\n",
    "\n",
    "data['hasBars '] = data['hasBars '].replace(mappingBars)\n",
    "data['hasBars '] = data['hasBars '].fillna(0)\n",
    "data['hasBars '] = data['hasBars '].astype(int)\n",
    "data['hasBars '].unique()\n",
    "\n",
    "\n",
    "mappingStorage = {\n",
    "    'אין מחסן': 0,\n",
    "    'יש מחסן': 1\n",
    "}\n",
    "\n",
    "data['hasStorage '] = data['hasStorage '].replace(mappingStorage)\n",
    "data['hasStorage '] = data['hasStorage '].fillna(0)\n",
    "data['hasStorage '] = data['hasStorage '].astype(int)\n",
    "data['hasStorage '].unique()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mappingHandicap = {\n",
    "    True: 1,\n",
    "    False: 0,\n",
    "    'לא': 0,\n",
    "    'כן': 1,\n",
    "    'no': 0,\n",
    "    'yes': 1,\n",
    "    'לא נגיש לנכים': 0,\n",
    "    'נגיש לנכים': 1,\n",
    "    'נגיש': 1,\n",
    "    'לא נגיש': 0,\n",
    "}\n",
    "\n",
    "data['handicapFriendly '] = data['handicapFriendly '].replace(mappingHandicap)\n",
    "data['handicapFriendly '] = data['handicapFriendly '].fillna(0)\n",
    "data['handicapFriendly '] = data['handicapFriendly '].astype(int)\n",
    "data['handicapFriendly '].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ניקיון נוסף וסידור לדאטה לפני בניית המודל  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>City</th>\n",
       "      <th>type</th>\n",
       "      <th>room_number</th>\n",
       "      <th>Area</th>\n",
       "      <th>Street</th>\n",
       "      <th>number_in_street</th>\n",
       "      <th>city_area</th>\n",
       "      <th>price</th>\n",
       "      <th>num_of_images</th>\n",
       "      <th>floor_out_of</th>\n",
       "      <th>...</th>\n",
       "      <th>hasBalcony</th>\n",
       "      <th>hasMamad</th>\n",
       "      <th>handicapFriendly</th>\n",
       "      <th>entranceDate</th>\n",
       "      <th>furniture</th>\n",
       "      <th>publishedDays</th>\n",
       "      <th>description</th>\n",
       "      <th>floor</th>\n",
       "      <th>total_floors</th>\n",
       "      <th>entrance_date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>5.5</td>\n",
       "      <td>137</td>\n",
       "      <td>רפאלי שרגא</td>\n",
       "      <td>3</td>\n",
       "      <td>אם המושבות החדשה</td>\n",
       "      <td>3600000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 11 מתוך 19</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>7</td>\n",
       "      <td>למכירה 5 5 חדרים ענקית מרווחת מוארת קומה גבוהה...</td>\n",
       "      <td>11</td>\n",
       "      <td>19</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>3</td>\n",
       "      <td>84</td>\n",
       "      <td>כצנלסון אהרון</td>\n",
       "      <td>6</td>\n",
       "      <td>נווה גן</td>\n",
       "      <td>2550000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 6 מתוך 9</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>מיידי</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>8</td>\n",
       "      <td>למכירה מפרטי ברחוב אהרון כצנלסון השקט והמבוקש ...</td>\n",
       "      <td>6</td>\n",
       "      <td>9</td>\n",
       "      <td>less_than_6_months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4</td>\n",
       "      <td>120</td>\n",
       "      <td>הירקונים</td>\n",
       "      <td>17</td>\n",
       "      <td>קרית הרב סלומון</td>\n",
       "      <td>2650000</td>\n",
       "      <td>10.0</td>\n",
       "      <td>קומה 2 מתוך 7</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>חלקי</td>\n",
       "      <td>6</td>\n",
       "      <td>פריים לוקשיין בשכונת שיפר המאוד מבוקשת למכירה ...</td>\n",
       "      <td>2</td>\n",
       "      <td>7</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>3.5</td>\n",
       "      <td>110</td>\n",
       "      <td>סלנט שמואל</td>\n",
       "      <td>56</td>\n",
       "      <td>המרכז השקט</td>\n",
       "      <td>2450000</td>\n",
       "      <td>8.0</td>\n",
       "      <td>קומה 2 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>חלקי</td>\n",
       "      <td>8</td>\n",
       "      <td>בפתח תקווה ברחוב שמואל סלנט המבוקשלא כדאי לפספ...</td>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>פתח תקווה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4.5</td>\n",
       "      <td>120</td>\n",
       "      <td>בן צבי יצחק</td>\n",
       "      <td>28</td>\n",
       "      <td>כפר גנים ב</td>\n",
       "      <td>2720000</td>\n",
       "      <td>9.0</td>\n",
       "      <td>קומה 3 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>21</td>\n",
       "      <td>הדירה משופצת חלקית בטעם טוב פונה לעורף לכיוון ...</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>694</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>בית פרטי</td>\n",
       "      <td>9.5 חד׳</td>\n",
       "      <td>350</td>\n",
       "      <td>הורד</td>\n",
       "      <td>35</td>\n",
       "      <td></td>\n",
       "      <td>8200000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קומה 4 מתוך 4</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>בית גדול מאוד 3 כיווני אוויר 2 מרפסות גדולות</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>695</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4 חד׳</td>\n",
       "      <td>110</td>\n",
       "      <td>קזן</td>\n",
       "      <td>NaN</td>\n",
       "      <td>מרכז דרום</td>\n",
       "      <td>3350000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 4 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>גמיש</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>בקומה 4 הדירה 110 מר נטו136 מר ארנונה מעוצבת ל...</td>\n",
       "      <td>4</td>\n",
       "      <td>6</td>\n",
       "      <td>flexible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>696</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>קוטג'</td>\n",
       "      <td>7 חד׳</td>\n",
       "      <td>376</td>\n",
       "      <td>הטללים</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קרית גנים</td>\n",
       "      <td>8500000</td>\n",
       "      <td>13.0</td>\n",
       "      <td>קומת קרקע</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2023-08-01 00:00:00</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>במערב המבוקש ה בית הנדיר הזה הוא עוד אחד מה הי...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>less_than_6_months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>697</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>5 חד׳</td>\n",
       "      <td>126</td>\n",
       "      <td>אחד העם</td>\n",
       "      <td>NaN</td>\n",
       "      <td>לסטר</td>\n",
       "      <td>3850000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>קומה 5 מתוך 7</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>מיידי</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>דירה חדשה מהקבלן באזור מבוקש 5 חדרים מרווחת ומ...</td>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>less_than_6_months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>698</th>\n",
       "      <td>רעננה</td>\n",
       "      <td>דירה</td>\n",
       "      <td>4.5 חד׳</td>\n",
       "      <td>140</td>\n",
       "      <td>קזן</td>\n",
       "      <td>10</td>\n",
       "      <td>מרכז דרום</td>\n",
       "      <td>3730000</td>\n",
       "      <td>6.0</td>\n",
       "      <td>קומה 3 מתוך 6</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2024-02-01 00:00:00</td>\n",
       "      <td>לא צויין</td>\n",
       "      <td>NaN</td>\n",
       "      <td>לא למתווכים ללא תיווך מתווכים לא להתקשר דירה ע...</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>months_6_12</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>693 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          City      type room_number  Area         Street number_in_street  \\\n",
       "0    פתח תקווה      דירה         5.5   137     רפאלי שרגא                3   \n",
       "1    פתח תקווה      דירה           3    84  כצנלסון אהרון                6   \n",
       "2    פתח תקווה      דירה           4   120       הירקונים               17   \n",
       "3    פתח תקווה      דירה         3.5   110     סלנט שמואל               56   \n",
       "4    פתח תקווה      דירה         4.5   120    בן צבי יצחק               28   \n",
       "..         ...       ...         ...   ...            ...              ...   \n",
       "694      רעננה  בית פרטי     9.5 חד׳   350           הורד               35   \n",
       "695      רעננה      דירה       4 חד׳   110            קזן              NaN   \n",
       "696      רעננה     קוטג'       7 חד׳   376         הטללים              NaN   \n",
       "697      רעננה      דירה       5 חד׳   126        אחד העם              NaN   \n",
       "698      רעננה      דירה     4.5 חד׳   140            קזן               10   \n",
       "\n",
       "            city_area    price  num_of_images     floor_out_of  ...  \\\n",
       "0    אם המושבות החדשה  3600000            6.0  קומה 11 מתוך 19  ...   \n",
       "1             נווה גן  2550000            6.0    קומה 6 מתוך 9  ...   \n",
       "2     קרית הרב סלומון  2650000           10.0    קומה 2 מתוך 7  ...   \n",
       "3          המרכז השקט  2450000            8.0    קומה 2 מתוך 6  ...   \n",
       "4          כפר גנים ב  2720000            9.0    קומה 3 מתוך 6  ...   \n",
       "..                ...      ...            ...              ...  ...   \n",
       "694                    8200000            NaN    קומה 4 מתוך 4  ...   \n",
       "695         מרכז דרום  3350000            6.0    קומה 4 מתוך 6  ...   \n",
       "696         קרית גנים  8500000           13.0        קומת קרקע  ...   \n",
       "697              לסטר  3850000            NaN    קומה 5 מתוך 7  ...   \n",
       "698         מרכז דרום  3730000            6.0    קומה 3 מתוך 6  ...   \n",
       "\n",
       "     hasBalcony   hasMamad   handicapFriendly         entranceDate   \\\n",
       "0            0.0          1                  1                 גמיש   \n",
       "1            0.0          1                  0                מיידי   \n",
       "2            1.0          1                  0                 גמיש   \n",
       "3            0.0          1                  1                 גמיש   \n",
       "4            1.0          1                  1                 גמיש   \n",
       "..           ...        ...                ...                  ...   \n",
       "694          1.0          1                  0                 גמיש   \n",
       "695          1.0          1                  1                 גמיש   \n",
       "696          0.0          0                  0  2023-08-01 00:00:00   \n",
       "697          1.0          1                  1                מיידי   \n",
       "698          1.0          1                  1  2024-02-01 00:00:00   \n",
       "\n",
       "    furniture   publishedDays   \\\n",
       "0     לא צויין               7   \n",
       "1     לא צויין               8   \n",
       "2         חלקי               6   \n",
       "3         חלקי               8   \n",
       "4     לא צויין              21   \n",
       "..         ...             ...   \n",
       "694          0             NaN   \n",
       "695   לא צויין             NaN   \n",
       "696   לא צויין             NaN   \n",
       "697   לא צויין             NaN   \n",
       "698   לא צויין             NaN   \n",
       "\n",
       "                                          description   floor  total_floors  \\\n",
       "0    למכירה 5 5 חדרים ענקית מרווחת מוארת קומה גבוהה...     11            19   \n",
       "1    למכירה מפרטי ברחוב אהרון כצנלסון השקט והמבוקש ...      6             9   \n",
       "2    פריים לוקשיין בשכונת שיפר המאוד מבוקשת למכירה ...      2             7   \n",
       "3    בפתח תקווה ברחוב שמואל סלנט המבוקשלא כדאי לפספ...      2             6   \n",
       "4    הדירה משופצת חלקית בטעם טוב פונה לעורף לכיוון ...      3             6   \n",
       "..                                                 ...    ...           ...   \n",
       "694       בית גדול מאוד 3 כיווני אוויר 2 מרפסות גדולות      4             4   \n",
       "695  בקומה 4 הדירה 110 מר נטו136 מר ארנונה מעוצבת ל...      4             6   \n",
       "696  במערב המבוקש ה בית הנדיר הזה הוא עוד אחד מה הי...      0             0   \n",
       "697  דירה חדשה מהקבלן באזור מבוקש 5 חדרים מרווחת ומ...      5             7   \n",
       "698  לא למתווכים ללא תיווך מתווכים לא להתקשר דירה ע...      3             6   \n",
       "\n",
       "          entrance_date  \n",
       "0              flexible  \n",
       "1    less_than_6_months  \n",
       "2              flexible  \n",
       "3              flexible  \n",
       "4              flexible  \n",
       "..                  ...  \n",
       "694            flexible  \n",
       "695            flexible  \n",
       "696  less_than_6_months  \n",
       "697  less_than_6_months  \n",
       "698         months_6_12  \n",
       "\n",
       "[693 rows x 26 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df= data.copy()\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ nan,  3. ,  4. ,  5. ,  6. ,  8. ,  2. ,  5.5,  3.5,  2.5,  6.5,\n",
       "        4.5,  7.5,  7. ,  1. ,  9. ,  9.5, 10. ])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# טיפול במספר חדרים ושינוי לעשרוני\n",
    "def room_number(x):\n",
    "    if isinstance(x, str):\n",
    "        x = re.sub(r'[^\\d\\.]', '', x)  # מחיקת תווים שאינם ספרות או פסיק\n",
    "        return x.replace(',', '')\n",
    "    elif isinstance(x, int):\n",
    "        return int(x) # חיזור ערך מספרי כמחרוזת\n",
    "    else:\n",
    "        return ''\n",
    "\n",
    "df['room_number']=df['room_number'].apply(lambda x: room_number(x)).replace('',np.nan).astype('float64')\n",
    "df['room_number'].unique()\n",
    "\n",
    "# טיפול בערך קיצוני- 35 חדרים\n",
    "df=df[df['room_number']!=35]\n",
    "df['room_number'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-15-84bb97cf5f3c>:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['City']= df['City'].apply(lambda x: x.strip()).replace('נהרייה','נהריה')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['פתח תקווה', 'נתניה', 'באר שבע', 'הרצליה', 'אריאל', 'דימונה',\n",
       "       'רחובות', 'גבעת שמואל', 'ירושלים', 'שוהם', 'כפר סבא', 'רעננה',\n",
       "       'נהריה', 'זכרון יעקב', 'קרית ביאליק', 'חיפה', 'הוד השרון',\n",
       "       'תל אביב', 'ראשון לציון', 'יהוד מונוסון', 'נס ציונה', 'אילת',\n",
       "       'חולון', 'מודיעין מכבים רעות', 'צפת', 'בת ים', 'רמת גן',\n",
       "       'נוף הגליל', 'בית שאן'], dtype=object)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['City']= df['City'].apply(lambda x: x.strip()).replace('נהרייה','נהריה')\n",
    "df['City'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-16-b709617b2f53>:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['condition '] = np.where((df['condition '] == 'None') | (df['condition '] == False), np.nan, df['condition '])\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['שמור', 'חדש', 'משופץ', 'לא צויין', 'ישן', 'דורש שיפוץ', nan],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#לסדר את עמודת מצב הנכס - condition\n",
    "df['condition '] = np.where((df['condition '] == 'None') | (df['condition '] == False), np.nan, df['condition '])\n",
    "df['condition '].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-17-10b9b30d5e50>:9: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['number_in_street'] = df['number_in_street'].apply(clean_number_in_street).replace('', np.nan)\n",
      "<ipython-input-17-10b9b30d5e50>:10: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df['number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce').astype('Int64')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<IntegerArray>\n",
       "[   3,    6,   17,   56,   28,    2,   18,    1,   23, <NA>,   27,    4,   71,\n",
       "    8,   37,    9,   29,   85,   46,   13,    7,   15,   26,   14,   35,   44,\n",
       "   36,   16,   34,   22,  123,  217,  105,   82,  131,  144,   42,   10,   12,\n",
       "   25,   53,   20,    5,   24,   40,   49,    0,   21,   59,   86,  121,   81,\n",
       "   78, 2003,   69,   11,   65,   57,   94,   32,  145,  221,   79,   19,   33,\n",
       "   52,   43,   50,   61,  110,   70,   63,   80,   68,   30,  342,   39,   41,\n",
       "  201, 2110,  200,  575,   72,  142,  264,  329,   47,  116,   58,   54,  212]\n",
       "Length: 91, dtype: Int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def clean_number_in_street(x):\n",
    "    if isinstance(x, str):\n",
    "        x = re.sub(r'[^\\d,]', '', x)  # מחיקת תווים שאינם ספרות או פסיק\n",
    "        return x.replace(',', '')  # הסרת פסיקים\n",
    "    elif isinstance(x, int):\n",
    "        return str(x)  # חיזור ערך מספרי כמחרוזת\n",
    "    else:\n",
    "        return ''\n",
    "df['number_in_street'] = df['number_in_street'].apply(clean_number_in_street).replace('', np.nan)\n",
    "df['number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce').astype('Int64')\n",
    "df['number_in_street'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#הורדת עמודות שהפכנו לעמודות אחרות\n",
    "df = df.drop(['floor_out_of','entranceDate ','publishedDays '],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna(subset = ['condition '])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "City                   0\n",
       "floor                  0\n",
       "description            0\n",
       "furniture              0\n",
       "handicapFriendly       0\n",
       "hasMamad               0\n",
       "hasBalcony             0\n",
       "condition              0\n",
       "hasStorage             0\n",
       "total_floors           0\n",
       "hasBars                0\n",
       "price                  0\n",
       "city_area              0\n",
       "Street                 0\n",
       "type                   0\n",
       "entrance_date          0\n",
       "hasElevator            1\n",
       "hasAirCondition        1\n",
       "hasParking             1\n",
       "num_of_images          8\n",
       "Area                  16\n",
       "room_number           62\n",
       "number_in_street     217\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum().sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(columns='description ', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df = df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df['Area'] = pd.to_numeric(df['Area'], errors='coerce')\n",
    "#df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')\n",
    "#df['number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "חלק 3 בניית המודל "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop(\"price\", axis=1)\n",
    "y = df.price.astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ניקח את כל הערכים החסרים בדאטה שלנו ונמלא אותם עם פונקציית סימפל אימפיוטר\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "columns_with_null = ['hasParking ', 'num_of_images', 'Area', 'room_number', 'number_in_street','hasElevator ','hasAirCondition ']\n",
    "\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "x[columns_with_null] = imputer.fit_transform(x[columns_with_null])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert x catergorical columns to dummies\n",
    "x_with_dummies = pd.get_dummies(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(x_with_dummies, y, test_size=0.20, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = x.select_dtypes(exclude=['object']).columns.tolist()\n",
    "cat_cols = x.select_dtypes(include=['object']).columns.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['City',\n",
       " 'type',\n",
       " 'Street',\n",
       " 'city_area',\n",
       " 'condition ',\n",
       " 'furniture ',\n",
       " 'entrance_date']"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cat_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['room_number',\n",
       " 'Area',\n",
       " 'number_in_street',\n",
       " 'num_of_images',\n",
       " 'hasElevator ',\n",
       " 'hasParking ',\n",
       " 'hasBars ',\n",
       " 'hasStorage ',\n",
       " 'hasAirCondition ',\n",
       " 'hasBalcony ',\n",
       " 'hasMamad ',\n",
       " 'handicapFriendly ',\n",
       " 'floor',\n",
       " 'total_floors']"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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/n4XFuJav4biy68yB5wudEwsAI6vluK+wGh6u3ZHFeajsoM1HetX4e5tNN0DrsDx5dJay3u790THNgod63PTcPxrRMj20Fs2oZJbo5JaoZwubGcOxFoTGKO1Zin2PpejlMLWoTsU56Nzs9A7tqI6nIgKi0DZHdi69MT65rMa26fefAnUM0tSH9o3ht1FZjHj0fDlLuiVHGjzc675oAf4JRcSwivPlXrNv3YbOGwms1OV0m2bsFq0Rjc317un70AcXwe2sePryr619vdtTha2n2svi9Y4vvgY93kXV9b35efoZi2x2rSvMf7jmgTG/9UOOLVZoM0veyvj37wX4iNqrLLOlG7bhDOhNc6EViiHi+iTBlL4fWBsEceejD3SXNfhrbvjycsw8TVKJLyVyXjawqNxJbXFk5dZL/3jmkBGoVnMeTWs2A49WwTabM6pjP3XHIiP9PnlgM5NYZnvw9+rodhdL3mOSYacfbA3D7wWbPwFulZZp5b71elymsweQGI8/LEb3L5s6m874PiauzqI5i0hdy/sywXLCz98Bx07Bdv1PA1+/hGKi/x8CYNj2sBGXxbR8kJZaf30/WkbD5kFsKfQtMU36XByq0CbMo+fvoPQg60etGoW2P7f/gRdQnxPOPNk+H6zydz58/sOKCn5cz78FWh16P4aisNtoWdXSs1WSv2glFqilIoAUErdopRao5T6Vin1P6VUpO/4YKXU977jafURUkqN8NW5SSk1wXfsUaXUnX4245VS/6eUilZKLVVKrVdKfaeUusxnMh1op5TaqJR6XBke9/n0nVLqGl89fZRSXyil3ga+q3er5GRCgt+naUISKicz2Kapv02yOQagQD1yM+reK+HjdwPbYe6TqOt7o5YvRP/zvmpdsGdl4k2qrN+bmIx9T6APtn256OhYcJiv6t6kQBvXdxtJvOZS4u8eimPrrwA4dm7HatyEuPGjaDrkcuImjkaVmBnClp2JlVipqZsmY8sO1FR5gZqWn01dylfF8dVSrIREbG1CzOCA3puJ8usLFZ+E3hvcFyrery+aJKNzMlHHdED/uBZdkIsuK8HakIbOyajRn8ON+HDI9puss0vNsero3wrW+a3NbcBT58Ab58PGPbB5X3AZT14mlt+1bDVNQlW91vZkogNsklF7MrHtqb2sfdNadON4dMs25kBJMa53ZlN2/d3VB+Ij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\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "correlations = X_train.loc[:,np.array(num_cols)].corr()\n",
    "plt.subplots(figsize=(10,10))\n",
    "sns.heatmap(correlations, cmap='cool', annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# we see strong connection between floor and total_floor probably because people are more willing to give their floor\n",
    "# number if they live on high floors\n",
    "x_with_dummies.drop(columns=['total_floors'], inplace=True)\n",
    "X_train.drop(columns=['total_floors'], inplace=True)\n",
    "X_test.drop(columns=['total_floors'], inplace=True)\n",
    "num_cols.remove('total_floors')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
