
## Analyze A/B Test Results


## Table of Contents
- [Introduction](#intro)
- [Part I - Probability](#probability)
- [Part II - A/B Test](#ab_test)
- [Part III - Regression](#regression)


<a id='intro'></a>
### Introduction

A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 

For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

<a id='probability'></a>
#### Part I - Probability

To get started, let's import our libraries.


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)
```

`1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**

a. Read in the dataset and take a look at the top few rows here:


```python
data = pd.read_csv("ab_data.csv")
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>851104</td>
      <td>2017-01-21 22:11:48.556739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>804228</td>
      <td>2017-01-12 08:01:45.159739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864975</td>
      <td>2017-01-21 01:52:26.210827</td>
      <td>control</td>
      <td>old_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



b. Use the below cell to find the number of rows in the dataset.


```python
data.shape[0]
```




    294478



c. The number of unique users in the dataset.


```python
uniqueUsers = data["user_id"].nunique()
uniqueUsers
```




    290584



d. The proportion of users converted.


```python
convertedUsers = data[data.converted == 1]["user_id"].nunique()
convertedUsers / uniqueUsers
```




    0.12104245244060237




```python
#Just checking here to see if what I did above is similar.
convertedUsers = data[data.converted == 1]["user_id"].shape[0]
convertedUsers / uniqueUsers
```




    0.12126269856564711



e. The number of times the `new_page` and `treatment` don't line up.


```python
notLinedUpA = data[(data.landing_page == "new_page")&(data.group != "treatment")]
notLinedUpB = data[(data.landing_page != "new_page")&(data.group == "treatment")]
notLinedUpA.shape[0] + notLinedUpB.shape[0]
```




    3893



f. Do any of the rows have missing values?


```python
data.isnull().values.any()
```




    False



`2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  

a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.


```python
#unAligned = data[((data.group =="treatment") & (data.landing_page != "new_page")) 
               #  | ((data.group =="control")&(data.landing_page != "old_page"))]
unAligned = data[((data['group'] == 'treatment') == (data['landing_page'] == 'new_page')) == False]
df2 = pd.concat([data, unAligned])
df2 = df2.drop_duplicates(keep=False)
```


```python
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
```




    0



<a href="https://stackoverflow.com/questions/37313691/how-to-remove-a-pandas-dataframe-from-another-dataframe">Got the deletion idea from here.</a>

`3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

a. How many unique **user_id**s are in **df2**?


```python
df2["user_id"].nunique()
```




    290584



b. There is one **user_id** repeated in **df2**.  What is it?


```python
df2[df2["user_id"].duplicated()].user_id
```




    2893    773192
    Name: user_id, dtype: int64



c. What is the row information for the repeat **user_id**? 


```python
df2[df2["user_id"].duplicated()]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2893</th>
      <td>773192</td>
      <td>2017-01-14 02:55:59.590927</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.


```python
#df2[df2.user_id == 773192]
df2 = df2.drop(df2.index[2893])
df2.shape[0]
```




    290584



`4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.

a. What is the probability of an individual converting regardless of the page they receive?


```python
probConversion = df2.converted.mean()
probConversion
```




    0.11959708724499628



b. Given that an individual was in the `control` group, what is the probability they converted?


```python
#df2[(df2.group == "control") & (df2.converted == 1)].shape[0] / df2[df2.group == "control"].shape[0]
df2[df2.group == "control"].converted.mean()
```




    0.1203863045004612



c. Given that an individual was in the `treatment` group, what is the probability they converted?


```python
 df2[df2.group == "treatment"].converted.mean()
```




    0.11880806551510564



d. What is the probability that an individual received the new page?


```python
probNewPage = df2[df2.landing_page == "new_page"].shape[0] / df2.shape[0]
probNewPage
```




    0.5000619442226688




```python
1 - probNewPage
```




    0.4999380557773312



e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

**At the moment, there doesn't seem to be enough evidence to support that either page is sufficiently better than the other. Let's put some data down.**

-  Probability of conversion: 11.9%
-  Probability of control group converting: 12%
-  Probability of treatment group converting: 11.9%

As we can see, control group has a minute advantage in converting to the new page, and even the overall conversion rate is pretty small. I won't say there's no evidence of conversion however - I can guess that sometimes, a 12% increase is substantial.

<a id='ab_test'></a>
### Part II - A/B Test

Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  

However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  

These questions are the difficult parts associated with A/B tests in general.  


`1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

null = Mean(new) - Mean(old) <= 0 <br>
alternate = Mean(new) - Mean(old) > 0

`2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>

Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>

Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>

Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

a. What is the **convert rate** for $p_{new}$ under the null? 


```python
pNew = np.random.binomial(1, probConversion, data.shape[0])
pNew.mean()
```




    0.11935017216905847




```python
p_new = df2[df2.converted == 1].shape[0]/(df2.shape[0])
p_new
```




    0.11959708724499628



b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

Since the question stated that we assume pnew and pold are equal, the answer is the same.



```python
pOld = pNew
pOld.mean()
```




    0.11935017216905847



c. What is $n_{new}$?


```python
nnew = df2[df2["landing_page"] == "new_page"].shape[0]
nnew
```




    145310



d. What is $n_{old}$?


```python
nold = df2[df2["landing_page"] == "old_page"].shape[0]
nold
```




    145274



e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.


```python
new_page_converted = np.random.binomial(1, pNew.mean(), nnew)
new_page_converted
```




    array([0, 0, 0, ..., 0, 0, 0])



f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.


```python
old_page_converted = np.random.binomial(1, pOld.mean(), nold)
old_page_converted
```




    array([0, 0, 0, ..., 0, 0, 1])



g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).


```python
new_page_converted.mean() - old_page_converted.mean()
```




    -0.0013511928759544073



h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.


```python
p_diffs = []
for x in range (10000):
    sample = data.sample(200, replace=True)
    new = sample[sample["landing_page"] == "new_page"].converted.mean()
    old = sample[sample["landing_page"] == "old_page"].converted.mean()
    p_diffs.append(new - old)
p_diffs = np.array(p_diffs)
```

i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.


```python
plt.hist(p_diffs);
```


![png](output_61_0.png)



```python
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals);
```


![png](output_62_0.png)



```python
convert_old = df2[df2['landing_page'] == "old_page"].converted == 1
convert_new = df2[df2['landing_page'] == "new_page"].converted == 1
n_old = df2.landing_page == "old_page"
n_new = df2.landing_page == "new_page"
prop_convert_new = convert_new.shape[0] / n_new.shape[0]
prop_convert_old = convert_old.shape[0] / n_old.shape[0]
```

j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?


```python
originalDifference = prop_convert_new - prop_convert_old
(null_vals > originalDifference).mean()
```




    0.49120000000000003



k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

**We just figured out the p-value of this dataset, which would tell us whether or not to refute the null hypothesis. With our margin of error for Type I errors at 5% (0.05), the probability of error is greater than that, which means we cannot refute the null, which is that the old page is just as good or better than the new one.**

l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.


```python
import statsmodels.api as sm
#Pushed the below stuff upstairs so no need to reinitialize.
#convert_old = df2[df2['landing_page'] == "old_page"].converted == 1
#convert_new = df2[df2['landing_page'] == "new_page"].converted == 1
#n_old = df2.landing_page == "old_page"
#n_new = df2.landing_page == "new_page"
```

    /opt/conda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.


```python
#I've used this, but I'm not sure if this is done correctly. 
#I think my end answer to (n) is still correct however
z_score, p_value = sm.stats.proportions_ztest([convert_old.shape[0], convert_new.shape[0]],[n_old.shape[0], n_new.shape[0]])

from scipy.stats import norm

# Tells us how significant our z-score is
print(norm.cdf(z_score))

#Tells us what our critical value at 95% confidence (5% error rate) is
print(norm.ppf(1-(0.05/2)))

z_score, p_value
```

    0.462377603979
    1.95996398454





    (-0.094445582555802571, 0.92475520795783672)



n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

**The Z-Score is smaller than the critical value, which means we cannot refute the null hypothesis, which states that the old page is just as good or better than the new pages. Likewise, the P-Value is at .92, which is a lot greater than 0.05, which means we cannot refute the null hypothesis, agreeing with our position from the earlier questions.**

<a id='regression'></a>
### Part III - A regression approach

`1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>

a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

**Logistic Regression, which predicts only two outcomes, in this case, conversion or no conversion.**

b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.


```python
import statsmodels.api as sm
import seaborn as sb

df2["intercept"] = 1
df2["ab_page"] = pd.get_dummies(df2.group == "treatment", drop_first = True)
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
      <th>intercept</th>
      <th>ab_page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>851104</td>
      <td>2017-01-21 22:11:48.556739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>804228</td>
      <td>2017-01-12 08:01:45.159739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864975</td>
      <td>2017-01-21 01:52:26.210827</td>
      <td>control</td>
      <td>old_page</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.


```python
model = sm.Logit(df2["converted"], df2[["intercept","ab_page"]])
results = model.fit()
```

    Optimization terminated successfully.
             Current function value: 0.366118
             Iterations 6


d. Provide the summary of your model below, and use it as necessary to answer the following questions.


```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290582</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     1</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>8.077e-06</td> 
</tr>
<tr>
  <th>Time:</th>              <td>00:44:31</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1899</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9888</td> <td>    0.008</td> <td> -246.669</td> <td> 0.000</td> <td>   -2.005</td> <td>   -1.973</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0150</td> <td>    0.011</td> <td>   -1.311</td> <td> 0.190</td> <td>   -0.037</td> <td>    0.007</td>
</tr>
</table>



e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>

**The p-value comes out to be 0.19. This p-value isn't describing the entire dataset, but the relationship between (in this case) the page and whether or not there was a conversion - hence the difference in values. To specify, the point of modeling this data is to see if there is a significant difference in conversion based on which page a customer receives. Thus, we can frame our H0 and H1 as:**

1. Null: The old page is better or as good at converting users.
2. Alt: The new page is better than the old page at converting users.

**Thus, we can see that since the p-value is 0.19 and above our usual 0.05% threshold, the regression model is saying that the alternate hypothesis is false (the page is not statistically significant), and that we cannot refute the null hypothesis.**

f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

**Depending on what data we have, sure, I can see why adding other elements might affect whether or not a user converts, and thus be useful in analysis. For example, perhaps the time of day matters, or whether or not this is a users first visit. The latter will have more weight, as someone who is new wouldn't care much for the change because he wouldn't know there was a change or have no trouble switching, while someone who has been using the site for a while will definitely be more hesitant to change - something I experienced very recently with Microsofts rebrand of VSTS to Azure Devops.**

g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 

Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.


```python
countries = pd.read_csv("countries.csv")
df2 = df2.set_index('user_id').join(countries.set_index('user_id'))
```


```python
df2.columns
```




    Index(['timestamp', 'group', 'landing_page', 'converted', 'intercept',
           'ab_page', 'country'],
          dtype='object')




```python
df2.country.value_counts()
dfC = df2
```


```python
df2[["countryA", "countryB"]] = pd.get_dummies(df2.country, drop_first = True)
```


```python
model2 = sm.Logit(df2["converted"], df2[["intercept","countryA", "countryB"]])
results2 = model2.fit()
results2.summary()
```

    Optimization terminated successfully.
             Current function value: 0.366116
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290581</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>1.521e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>00:44:32</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1984</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -2.0375</td> <td>    0.026</td> <td>  -78.364</td> <td> 0.000</td> <td>   -2.088</td> <td>   -1.987</td>
</tr>
<tr>
  <th>countryA</th>  <td>    0.0507</td> <td>    0.028</td> <td>    1.786</td> <td> 0.074</td> <td>   -0.005</td> <td>    0.106</td>
</tr>
<tr>
  <th>countryB</th>  <td>    0.0408</td> <td>    0.027</td> <td>    1.518</td> <td> 0.129</td> <td>   -0.012</td> <td>    0.093</td>
</tr>
</table>



**Looking at the data, we again see that the p-values for the countries are greater than alpha, and thus statistically insignificant. To be safe, I have tried to use country C as well, with the same results.**


```python
dfC[["countryA", "countryB", "countryC"]] = pd.get_dummies(df2.country)
dfC = dfC.drop("countryB", 1)
modelC = sm.Logit(df2["converted"], df2[["intercept","countryA", "countryC"]])
resultsC = modelC.fit()
resultsC.summary()
```

    Optimization terminated successfully.
             Current function value: 0.366116
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290581</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>1.521e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>00:44:33</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1984</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9868</td> <td>    0.011</td> <td> -174.174</td> <td> 0.000</td> <td>   -2.009</td> <td>   -1.964</td>
</tr>
<tr>
  <th>countryA</th>  <td>   -0.0507</td> <td>    0.028</td> <td>   -1.786</td> <td> 0.074</td> <td>   -0.106</td> <td>    0.005</td>
</tr>
<tr>
  <th>countryC</th>  <td>   -0.0099</td> <td>    0.013</td> <td>   -0.746</td> <td> 0.456</td> <td>   -0.036</td> <td>    0.016</td>
</tr>
</table>



h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  

Provide the summary results, and your conclusions based on the results.


```python
model3 = sm.Logit(df2["converted"], df2[["intercept", "ab_page", "countryA", "countryB"]])
result3 = model3.fit()
result3.summary()
```

    Optimization terminated successfully.
             Current function value: 0.366113
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290580</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     3</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>2.323e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>00:44:34</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1760</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -1.9893</td> <td>    0.009</td> <td> -223.763</td> <td> 0.000</td> <td>   -2.007</td> <td>   -1.972</td>
</tr>
<tr>
  <th>ab_page</th>   <td>   -0.0149</td> <td>    0.011</td> <td>   -1.307</td> <td> 0.191</td> <td>   -0.037</td> <td>    0.007</td>
</tr>
<tr>
  <th>countryA</th>  <td>   -0.0408</td> <td>    0.027</td> <td>   -1.516</td> <td> 0.130</td> <td>   -0.093</td> <td>    0.012</td>
</tr>
<tr>
  <th>countryB</th>  <td>    0.0099</td> <td>    0.013</td> <td>    0.743</td> <td> 0.457</td> <td>   -0.016</td> <td>    0.036</td>
</tr>
</table>



### Conclusions

**Looking at the above results, the country doesn't seem to have any significance on conversion. Overall, it seems that the new and old page have little effect on conversion. This means, unless there's a specific reason to use the new page, threre isn't enough evidence to deploy it since the old page performs just as well.**

We looked at a few conversion rates:

-  Probability of conversion: 11.9%
-  Probability of control group converting: 12%
-  Probability of treatment group converting: 11.9%

We looked at the differences between our boostrapped differences vs the population differences and found little to separate them:



```python
plt.hist(p_diffs, alpha=0.5);
plt.hist(null_vals, alpha=0.5);
```


![png](output_98_0.png)


We worked through the same examples through Logistic Regression, this time including country to see if that had any effect:


```python
results2.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>converted</td>    <th>  No. Observations:  </th>   <td>290584</td>   
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>290581</td>   
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>   
</tr>
<tr>
  <th>Date:</th>          <td>Sat, 05 Jan 2019</td> <th>  Pseudo R-squ.:     </th>  <td>1.521e-05</td> 
</tr>
<tr>
  <th>Time:</th>              <td>00:48:37</td>     <th>  Log-Likelihood:    </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td>-1.0639e+05</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>   <td>0.1984</td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   -2.0375</td> <td>    0.026</td> <td>  -78.364</td> <td> 0.000</td> <td>   -2.088</td> <td>   -1.987</td>
</tr>
<tr>
  <th>countryA</th>  <td>    0.0507</td> <td>    0.028</td> <td>    1.786</td> <td> 0.074</td> <td>   -0.005</td> <td>    0.106</td>
</tr>
<tr>
  <th>countryB</th>  <td>    0.0408</td> <td>    0.027</td> <td>    1.518</td> <td> 0.129</td> <td>   -0.012</td> <td>    0.093</td>
</tr>
</table>



With the P-Value > 0.5 , we identified that this isn't particularly significant - that is, the page doesn't have any significance on whether or not a conversion happened.

<b>Thus, we can conclude that conversion is independent of these factors, and our null hypothesis - the old page is just as good or better than the new page at conversion - cannot be refuted.<b> 
<br>
<b>I'd suggest that unless there are other business reasons, there aren't any practical reasons to spend time or money on deploying this new page. If a 12% conversion rate - over 11 - is of absolute significance, then perhaps it's viable to do so. <b>
<br>


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])
```




    0




```python

```
