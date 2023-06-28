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
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#ניקח את כל הערכים החסרים בדאטה שלנו ונמלא אותם עם פונקציית סימפל אימפיוטר\n",
    "from sklearn.impute import SimpleImputer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# preprocessing steps: 1. standard scaler, 2. one hot encoding\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train[num_cols])\n",
    "X_train[num_cols] = scaler.transform(X_train[num_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "10it [00:17,  1.79s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "alpha=0.1, l1_ratio=0.9\n",
      "mse: 1364998221762.6616\n",
      "rmse: 1168331.3835392173\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import KFold\n",
    "from sklearn.linear_model import ElasticNet\n",
    "from tqdm import tqdm\n",
    "kf = KFold(n_splits=10, random_state=42, shuffle=True)\n",
    "results = dict()\n",
    "for i, (train_index, test_index) in tqdm(enumerate(kf.split(X_train, y_train))):\n",
    "    for alpha in (0.1, 0.2, 0.4, 0.6, 0.8, 1):\n",
    "        for l1_ratio in (0.2, 0.5, 0.7, 0.8, 0.9):\n",
    "            elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)\n",
    "            elastic_net.fit(X_train.iloc[train_index], y_train.iloc[train_index])\n",
    "            predictions = elastic_net.predict(X_train.iloc[test_index])\n",
    "            mse_loss = np.mean((predictions - y_train.iloc[test_index])**2)\n",
    "            if f'alpha={alpha}, l1_ratio={l1_ratio}' in results:\n",
    "                results[f'alpha={alpha}, l1_ratio={l1_ratio}'] += mse_loss\n",
    "            else:\n",
    "                results[f'alpha={alpha}, l1_ratio={l1_ratio}'] = mse_loss\n",
    "best_params = min(results, key=results.get)\n",
    "loss_of_best = results[best_params]/10\n",
    "print(best_params)\n",
    "print('mse:', loss_of_best)\n",
    "print('rmse:', np.sqrt(loss_of_best))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# preprocessing test\n",
    "X_test[num_cols] = scaler.transform(X_test[num_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1108186457391810.1\n",
      "33289434.621089768\n"
     ]
    }
   ],
   "source": [
    "elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.9, random_state=42)\n",
    "elastic_net.fit(X_train, y_train)\n",
    "predictions = elastic_net.predict(X_test)\n",
    "mse_loss = np.mean((predictions - y_test)**2)\n",
    "rmse_loss = np.sqrt(mse_loss)\n",
    "print(mse_loss)\n",
    "print(rmse_loss)"
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
