# LOGISTIC REGRESSION 

Example 1:Personal loan offer
outcome var = accept banl loan(0/1)
predictors = Demographic info , and info about their bank relationship

Single Predictor model
PS:based on income what is the prob of getting  loan
dv=loan
idv=income


```python
import pandas as pd
```


```python
dataset = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)
```


```python
dataset.columns
```




    Index(['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
           'Education', 'Mortgage', 'Personal Loan', 'Securities Account',
           'CD Account', 'Online', 'CreditCard'],
          dtype='object')




```python
dataset1=dataset.dropna()
```


```python
dataset2=dataset1.drop_duplicates()
```


```python
dataset3=dataset2.drop(['ID','ZIP Code'],axis=1) #coz this are not used since they are categorical
```


```python
y=dataset3["Personal Loan"]
```


```python
x=dataset3.drop(['Personal Loan'],axis=1)  
x.head()#coz dv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Experience</th>
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>1</td>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>19</td>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>15</td>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>9</td>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
import statsmodels.api as sm
x1=sm.add_constant(x)
Logistic=sm.Logit(y,x1)
result=Logistic.fit()
result.summary()

```

    Optimization terminated successfully.
             Current function value: 0.128435
             Iterations 9
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Personal Loan</td>  <th>  No. Observations:  </th>  <td>  5000</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  4988</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    11</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 21 Jan 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.5938</td> 
</tr>
<tr>
  <th>Time:</th>                <td>19:35:36</td>     <th>  Log-Likelihood:    </th> <td> -642.18</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -1581.0</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>              <td>  -12.1928</td> <td>    1.645</td> <td>   -7.411</td> <td> 0.000</td> <td>  -15.417</td> <td>   -8.968</td>
</tr>
<tr>
  <th>Age</th>                <td>   -0.0536</td> <td>    0.061</td> <td>   -0.874</td> <td> 0.382</td> <td>   -0.174</td> <td>    0.067</td>
</tr>
<tr>
  <th>Experience</th>         <td>    0.0638</td> <td>    0.061</td> <td>    1.046</td> <td> 0.295</td> <td>   -0.056</td> <td>    0.183</td>
</tr>
<tr>
  <th>Income</th>             <td>    0.0546</td> <td>    0.003</td> <td>   20.831</td> <td> 0.000</td> <td>    0.049</td> <td>    0.060</td>
</tr>
<tr>
  <th>Family</th>             <td>    0.6958</td> <td>    0.074</td> <td>    9.364</td> <td> 0.000</td> <td>    0.550</td> <td>    0.841</td>
</tr>
<tr>
  <th>CCAvg</th>              <td>    0.1240</td> <td>    0.040</td> <td>    3.127</td> <td> 0.002</td> <td>    0.046</td> <td>    0.202</td>
</tr>
<tr>
  <th>Education</th>          <td>    1.7362</td> <td>    0.115</td> <td>   15.088</td> <td> 0.000</td> <td>    1.511</td> <td>    1.962</td>
</tr>
<tr>
  <th>Mortgage</th>           <td>    0.0005</td> <td>    0.001</td> <td>    0.856</td> <td> 0.392</td> <td>   -0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>Securities Account</th> <td>   -0.9368</td> <td>    0.286</td> <td>   -3.277</td> <td> 0.001</td> <td>   -1.497</td> <td>   -0.377</td>
</tr>
<tr>
  <th>CD Account</th>         <td>    3.8225</td> <td>    0.324</td> <td>   11.800</td> <td> 0.000</td> <td>    3.188</td> <td>    4.457</td>
</tr>
<tr>
  <th>Online</th>             <td>   -0.6752</td> <td>    0.157</td> <td>   -4.298</td> <td> 0.000</td> <td>   -0.983</td> <td>   -0.367</td>
</tr>
<tr>
  <th>CreditCard</th>         <td>   -1.1197</td> <td>    0.205</td> <td>   -5.462</td> <td> 0.000</td> <td>   -1.522</td> <td>   -0.718</td>
</tr>
</table>




```python

```
