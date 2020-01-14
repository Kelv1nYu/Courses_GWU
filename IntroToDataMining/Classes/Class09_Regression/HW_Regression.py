#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

#%%
# Q1
Data = pd.read_csv("bikedata.csv")
bikeData = Data[(Data['Working Day'] == 1) & (Data['Hour'] == 13) & (Data['Day of the Week'] <= 5) & (Data['Day of the Week'] >=1)]
print(bikeData.head())


# %%
# Q2
# Use Season, Temperature F, Humidity, Wind Speed
bikeData.columns = ['Date', 'Season', 'Hour', 'Holiday', 'DotW', 'WD', 'WT', 'TF', 'TFF', 'Hum', 'WS', 'CU', 'RU', 'TU']
modelBike = ols(formula = 'CU ~ TF + Hum + WS + C(Season)', data = bikeData).fit()
print(modelBike.summary())

# %%
# Q3
# Since the summary of last model shows that the Wind Speed has p-value 0.073 > 0.05, which means that this variable's coefficient is not significant, so we drop it
# Although the Season 3 also has p-value 0.624 > 0.05, coefficients of other parts are significant, so we keep them in.
bikeData.columns = ['Date', 'Season', 'Hour', 'Holiday', 'DotW', 'WD', 'WT', 'TF', 'TFF', 'Hum', 'WS', 'CU', 'RU', 'TU']
modelBike = ols(formula = 'CU ~ TF + Hum + C(Season)', data = bikeData).fit()
print(modelBike.summary())

#%%
# Q4
interModelBike = ols(formula = 'CU ~ C(Season) : (TF + Hum)', data = bikeData).fit()
print(interModelBike.summary())

#%%
# Q5
# import data
titanic = pd.read_csv("Titanic_clean.csv")

#%%
# Q5.1
ageTitanic = titanic.age
maleAge = ageTitanic[titanic.sex == 'male']
femaleAge = ageTitanic[titanic.sex == 'female']


ax1 = maleAge.plot(x = 'age', kind = 'hist', color = 'red', label = 'male')
femaleAge.plot(x = 'age', kind = 'hist', color = 'blue', label = 'female')

plt.legend(loc = 'upper right')
plt.title("Histogram on age")
plt.show()

#%%
# Q5.2
sex = titanic.sex
male = sex[titanic.sex == 'male']
female = sex[titanic.sex == 'female']
maleSum = len(male)
femaleSum = len(female)
malePre = maleSum / len(sex)
femalePre = femaleSum / len(sex)

sur = titanic.survived
surSum = len(sur[titanic.survived == 1])
unsurSum = len(sur[titanic.survived == 0])
surPre = surSum / len(sur)
unsurPre = unsurSum / len(sur)

maleSurSum = len(male[titanic.survived == 1])
maleUnsurSum = len(male[titanic.survived == 0])
maleSurInMale = maleSurSum / maleSum
maleSurInSur = maleSurSum / surSum
maleUnsurInMale = maleUnsurSum / maleSum
maleUnsurInSur = maleUnsurSum / unsurSum
femaleSurSum = len(female[titanic.survived == 1])
femaleUnsurSum = len(female[titanic.survived == 0])
femaleSurInFemale = femaleSurSum / femaleSum
femaleSurInSur = femaleSurSum / surSum
femaleUnsurInFemale = femaleUnsurSum / femaleSum
femaleUnsurInSur = femaleUnsurSum / unsurSum

print("In this DataSet, We can find that there are %d people, %.2f %% of them are male, %.2f %% of them are female." % (len(sex), malePre, femalePre))
print("there are %.2f %% people survival, %.2f %% of them are male, %.2f %% of them are female." % (surPre, maleSurInSur, femaleSurInSur))
print("there are %.2f %% people nonsurvival, %.2f %% of them are male, %.2f %% of them are female." % (unsurPre, maleUnsurInSur, femaleUnsurInSur))
print("In other word, %.2f %% of male are survived, %.2f %% are not survived, %.2f %% of female are survived, %.2f %% are not survived." % (maleSurInMale, maleUnsurInMale, femaleSurInFemale, femaleUnsurInFemale))


#%%
# Q5.3

titanic.pclass.groupby(titanic.pclass).count().plot(kind = 'pie', colors = ['red', 'orange', 'green'], autopct='%.2f %%')
plt.title("pie chart for Ticket class")
plt.show()

#%%
# Q6
# Use age, sex, pclass, sibsp as feature, parch, ticket, fare, embarked are not good features
modelLogit = glm(formula='survived ~ age + C(sex) + C(pclass) + sibsp', data=titanic, family=sm.families.Binomial()).fit()
print(modelLogit.summary())

#%% [markdown]

# ###Q7

# * Through the analysis and modeling of the data set, I found that the probability of survival of the passengers on the Titanic was related to variables age, gender, pclass and sibsp.

# * The coef of age is -0.0448, which means that the survival rate will decrease by 0.037 for every 1-year-old increase, so there may be more children in the survivors.

# * In the judgment of gender, the summary of model shows that male has a negative impact on the probability of survival (-2.6277), it indicates that there are more female survivors.

# * According to the analysis of pclass, with the decrease of its level, the probability of survival will also decrease, that is to say, the higher the pclass level, the more people survive (most of them are people with money or status at that time).

# * For extra tests and model, sibsp also have effect on survival(I don't know why, but the result of test shows it that when the number of siblings / spouses increase one unit, the survival rate will decrease by 0.3802).

#%%
# Q8

def cal(cut_off):
    # Compute class predictions
    titanic['survived_Logit'] = modelLogit.predict(titanic)
    titanic['classLogitAll'] = np.where(titanic['survived_Logit'] > cut_off, 1, 0)
    # Make a cross table
    crossTable = pd.crosstab(titanic.survived, titanic.classLogitAll,
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)
    accuracy = (crossTable.loc[1, 1] + crossTable.loc[0, 0]) / crossTable.iloc[2, 2]
    precision = crossTable.loc[1, 1] / (crossTable.loc[1, 1] + crossTable.iloc[0, 1])
    recallRate = crossTable.loc[1, 1] / (crossTable.loc[1, 1] + crossTable.iloc[1, 0])
    return accuracy, precision, recallRate

ac1, pr1, re1 = cal(0.3)
ac2, pr2, re2 = cal(0.5)
ac3, pr3, re3 = cal(0.7)


print("when the cut_off = 0.3, Total accuracy of the model is %.2f, The precision of the model is %.2f, The recall rate is %.2f" % (ac1, pr1, re1))
print("when the cut_off = 0.5, Total accuracy of the model is %.2f, The precision of the model is %.2f, The recall rate is %.2f" % (ac2, pr2, re2))
print("when the cut_off = 0.7, Total accuracy of the model is %.2f, The precision of the model is %.2f, The recall rate is %.2f" % (ac3, pr3, re3))





# %%
