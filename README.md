# Code Scripts and Data for KDD'21 paper: MTC

### Cite
contact Chaoqi Yang (chaoqiy2@illinois.edu, ycqsjtu@gmail.com) for any question.

```bibtex
@inproceedings{yang2021mtc,
    title = {MTC: Multiresolution Tensor Completion from Partial and Coarse Observations},
    author = {Yang, Chaoqi and Singh, Navjot and Xiao, Cao and Solomonik, Edgar and Sun, Jimeng},
    booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) 2021},
    year = {2021}
}
```

We provide the code for (1) tensor data processing; (2) baselines; (3) our models.
## 0. Quick start
Run our model MTC on the synthetic data
```python
cd src_synthetic_numpy; python test_X_C1_C2_unknown.py --partial 0.05 --jacobi 1 --multi 1
```

Run our model MTC on a subset of GCSS data
```python
cd src_GCSS_torch; python run.py --model MTC --partial 0.05 --jacobi 1 --multi 1
```


## 1. Folder Tree

- ./data_synthetic
    - the generated synthetic data (.pkl format): X, C1, C2, P1, P2
    - the generating script: synthetic.ipynb
- ./src_synthetic_numpy (numpy implementation for synthetic data)
    * codes with only partial observation: **test_only_T.py**
    * codes with X, C1, C2 and unknown P1, P2: **test_X_C1_C2_unknown.py**
    * codes with X, C1, C2 and known P1, P2: **test_X_C1_C2_known.py**
- ./data_GCSS
    - data download link: https://pair-code.github.io/covid19_symptom_dataset/?country=GB (The US data)
    - a subset of GCSS data (.pkl format): X, O1, O2, P1, P2
    - data preprocessing jupyter notebook: **Google Tensor Data.ipynb**
- ./src_GCSS_numpy (numpy implementation for GCSS data)
    - MTC codes with X, O1, O2 and unknown P1, P2: **X_O1_O2_unknwon_P1_unknown_P2.py**
    - Baseline codes with O1, O2 and unknown P1, P2: **OracleCPD.py, BGD.py, PREMA.py, CMTF.py**
    - MTC codes with X, O1, O2 and unknown P1, known P2: **X_O1_O2_unknwon_P1_known_P2.py**
    - Codes for tensor initialization comparison: **initialization.py**
- ./src_GCSS_torch (torch implementation for GCSS data, <u>**GPU support**</u>)
    - the interfact to call completion models: run.py
    - different model functions: model.py

## 2. Python package dependency

```bash
pip install scipy==1.5.0
pip install numpy==1.19.1
pip install pandas==1.1.2
pip install torch==1.7.0
```
The dependency packages are pretty common. If any missing, please install that.

## 3. Instruction for synthetic data
### 3.1 Generate the synthetic data
- The following is the key scripts to generate the synthetic data by **./src_synthetic/synthetic.ipynb**.
    ```python
    import numpy as np
    np.random.seed(400)
    # generate factors
    U = np.random.random((125, 10))
    V = np.random.random((125, 10))
    W = np.random.random((125, 10))
    U = np.sort(U, axis=0) # make it smooth, so as to be continuous mode
    W = np.sort(W, axis=0) # make it smooth, so as to be continuous mode

    # tensor X
    T = np.einsum('ir,jr,kr->ijk',U,V,W,optimize=True)

    # mapping on the first mode
    P1 = np.random.random((12, 125))
    P1 = np.argsort(P1, axis=0) <= 0.0

    # mapping on the first mode
    P2 = np.random.random((12, 125))
    P2 = np.argsort(P2, axis=0) <= 0.0

    # C1, C2
    C1 = np.einsum('ijk,ri->rjk',T,P1,optimize=True)
    C2 = np.einsum('ijk,rj->irk',T,P2,optimize=True)
    ```

- We also provide the data (used in the paper): **./data_synthetic**.

### 3.2 Run and visualize the experiments
The following is the argument options:
```
optional arguments:
  -h, --help         show this help message and exit
  --partial PARTIAL  the amount of partial observation
  --jacobi JACOBI    Jacobi (1) or not (0)
  --multi MULTI      multiresolution (1) or not (0)
```
To get the results with only the partial observation, use
```python
python test_only_X.py --partial [0.05 or other] --jacobi [0/1] --multi [0/1]
```
To get the results with also coarse information (C1, C2), but not with the aggregation (P1, P2), use
```python
python test_X_C1_C2_unknown.py --partial [0.05 or other] --jacobi [0/1] --multi [0/1]
```

To get the results with also coarse information (C1, C2) and the aggregation (P1, P2), use
```python
python test_X_C1_C2_known.py --partial [0.05 or other] --jacobi [0/1] --multi [0/1]
```
In fact, we also provide all the running result files in **./src_synthetic_numpy/synthetic_result/..**, and the results could be fed into the provided jupyter notebook for visualization.

## 4. Instruction for GCSS data
### 4.1 data preprocessing
- We already provide a subset of GCSS data in **./data_GCSS**. To get the whole tensor, following these steps:
    - First, download the GCSS data from https://pair-code.github.io/covid19_symptom_dataset/?country=GB
    - Second, run the data process script (Google Tensor Data.ipynb) to generate X, C1, C3, P1, P3 (Since in this dataset, the aggregation is on the first and the third mode, the C1, C3 are coarse tensors, and P1, P3 are the according aggregation matrices), or using the following code equivalently,
    ```python
    import pandas as pd
    import numpy as np

    data = pd.read_csv('./2020_country_daily_2020_US_daily_symptoms_dataset.csv') # change the data path accordingly

    # all zipcode
    zipcode = list(filter(lambda x: ('US' in x) and (x.count('_') == 2), data.key.unique()))
    # the time
    time = data.date.unique().tolist()
    # disease list
    disease = data.columns.tolist()[2:]

    # tensor placeholder
    T = np.zeros((len(zipcode), len(disease), len(time)))

    # process per zipcode
    for i, single_zip in enumerate(zipcode):
        print (i, single_zip)
        tmp = data[data.key == single_zip]
        re_time = [time.index(i) for i in tmp.date.tolist()]
        T[i, :, :][:, re_time] = tmp.values[:, 2:].T

    # change nan to zero
    T = np.nan_to_num(T)

    # zipcode -> state mapping
    state = np.unique([item.split('_')[1] for item in zipcode]).tolist()
    P1 = np.zeros((len(state), len(zipcode)))

    for i, z in enumerate(zipcode):
        for j, s in enumerate(state):
            if s in z:
                P1[j, i] = 1
                
    # date -> week mapping
    P3 = np.zeros(((len(time) - 1) // 7 + 1, len(time)))
    for i, t in enumerate(time):
        P3[i // 7, i] = 1

    O1 = np.einsum('ijk,ri->rjk',T,P1,optimize=True)
    O3 = np.einsum('ijk,rk->ijr',T,P3,optimize=True)

    import pickle 
    # dump
    pickle.dump(O1, open('O1_google.pkl', 'wb'))
    pickle.dump(O3, open('O2_google.pkl', 'wb'))
    pickle.dump(T, open('X_google.pkl', 'wb'))
    pickle.dump(P1, open('P1_google.pkl', 'wb'))
    pickle.dump(P3, open('P2_google.pkl', 'wb'))
    ```

- ATTENTION! In the code, the variable names might be a little bit different, for example, we use variable name O1 to denote the tensor aggregated on the first mode. Take care, not get confused.

### 4.2 Use torch implementation
One command for all:

```python
python run.py --model [model name] --partial [0.05 or other] --jacobi [0/1] --multi [0/1]
```
The [model name] could be (1) ```OracleCPD```; (2) ```CPC-ALS```; (3) ```BGD```; (4) ```CMTF```; (5) ```PREMA```; (6) ```MTC```; (7) ```MTC-P2```. Here, ```MTC``` is our proposed model without knowning P1,P2 while ```MTC-P2``` knows P2 but not P1. Also, the last two arguments are only for ```MTC``` and ```MTC-P2```.

### 4.3 Use numpy implementation
Oracle CPD model: run standard CP decomposition model on the origianl tensor
```python
python OracleCPD.py
```

CPC-ALS model: only use the partial observation, treat the problem as a tensor completion problem
```python
python CPC_ALS.py --partial [0.05 or other]
```

For BGD, CMTF, PREMA
```python
python BGD.py --partial [0.05 or other]
python CMTF.py --partial [0.05 or other]
python PREMA.py --partial [0.05 or other]
```
For our model variant: MTC_{both-}, MTC_{multi-}, MTC_{jacobi-}, MTC:
```python
python X_C1_C3_unknwon_P1_unknown_P3.py --partial [0.05 or other] --jacobi [0/1] --multi [0/1]
```
