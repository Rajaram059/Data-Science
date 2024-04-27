#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('covid_19')
df.head()


# In[3]:


df=df.rename(columns = {'Province/State' : 'state',
                        'Country/Region':'country',
                        'Lat':'lat',
                        'Long':'long',
                        'Date':'date',
                        'Confirmed':'confirmed',
                        'Deaths':'deaths',
                        'Recovered':'recovered',
                        'Active':'active',
                        'WHO Region':'who' } )


# In[4]:


df.head()


# In[5]:


df['active'] = df['confirmed'] - df['deaths']-df['recovered']


# In[6]:


df['active']


# In[7]:


top = df[df['date']==df['date'].max()]


# In[8]:


top


# In[9]:


c = top.groupby('country')['confirmed','active','deaths'].sum().reset_index()


# In[10]:


c


# In[14]:


fig = px.choropleth(c,locations = 'country',locationmode = 'country names',color = 'deaths',hover_name = 'country',
                   range_color = [1,1500],color_continuous_scale = "peach",title = 'Active cases Countries')


# In[15]:


fig.show()


# In[20]:


plt.figure(figsize = (15,10))
t_cases = df.groupby('date')['confirmed'].sum().reset_index()
t_cases['date'] = pd.to_datetime(t_cases['date'])
a = sns.pointplot(x = t_cases.date.dt.date,
                  y = t_cases.confirmed,
                  color = 'r')
a.set(xlabel = 'dates' ,ylabel = 'total cases')
plt.xticks(rotation = 90,fontsize = 5)
plt.yticks(fontsize = 15)
plt.show()


# In[30]:


t_casesactive = top.groupby(by= 'country')['active'].sum().sort_values(ascending = False).head(20).reset_index() 


# In[31]:


t_casesactive


# In[42]:


plt.figure(figsize = (15,10))
plt.title('Top 20 countries having most active cases',fontsize = 20)
a = sns.barplot(x = t_casesactive.active,
                y = t_casesactive.country)
for i,(value,name) in enumerate(zip(t_casesactive.active,t_casesactive.country)):
    a.text(value,i-.05,f'{value:,.0f}',size=10,ha='left',va='center')
a.set(xlabel='Cases total',ylabel='Country')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Cases total',fontsize=20)
plt.ylabel('Countryl',fontsize=20)


# In[34]:


t_casesdeath = top.groupby(by= 'country')['deaths'].sum().sort_values(ascending = False).head(20).reset_index() 


# In[35]:


t_casesdeath


# In[41]:


plt.figure(figsize = (15, 10))
plt.title("Top 20 countries having most death cases", fontsize = 30)
a = sns.barplot(x = t_casesdeath.deaths,
                y = t_casesdeath.country)

for i,(value, name) in enumerate(zip(t_casesdeath.deaths, t_casesdeath.country)):
    a.text(value, i-.05, f'{value:,.0f}', size = 10, ha = 'left', va = 'center')

a.set(xlabel='Total Death Cases',ylabel='Country')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Total Death Cases',fontsize=20)
plt.ylabel('Countryl',fontsize=20)


# In[43]:


t_casesrecover = top.groupby(by='country')['recovered'].sum().sort_values(ascending=False).head(20).reset_index()
t_casesrecover


# In[44]:


plt.figure(figsize = (15, 10))
plt.title("Top 20 countries having most Recovered cases", fontsize = 30)
a = sns.barplot(x = t_casesrecover.recovered,
                y = t_casesrecover.country)

for i,(value, name) in enumerate(zip(t_casesrecover.recovered, t_casesrecover.country)):
  a.text(value, i-.05, f'{value:,.0f}', size = 10, ha = 'left', va = 'center')

a.set(xlabel='Total Death Cases',ylabel='Country')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Total Death Cases',fontsize=20)
plt.ylabel('Countryl',fontsize=20)


# In[45]:


Brazil = df[df.country =='Brazil']
Brazil = Brazil.groupby(by='date')['recovered', 'active', 'deaths', 'confirmed'].sum().reset_index()
Brazil.tail(20)


# In[46]:


US = df[df.country =='US']
US = US.groupby(by='date')['recovered', 'active', 'deaths', 'confirmed'].sum().reset_index()
US.tail(20)


# In[47]:


Russia = df[df.country =='Russia']
Russia = Russia.groupby(by='date')['recovered', 'active', 'deaths', 'confirmed'].sum().reset_index()
Russia.tail(20)


# In[48]:


India = df[df.country =='India']
India = India.groupby(by='date')['recovered', 'active', 'deaths', 'confirmed'].sum().reset_index()
India.tail(20)


# In[49]:


sns.pointplot(data = Brazil, x = Brazil.index, y= 'confirmed', color = 'green', label = 'Brazil')
sns.pointplot(data = US, x = US.index, y= 'confirmed', color = 'red', label = 'US')
sns.pointplot(data = Russia, x = Russia.index, y= 'confirmed', color = 'blue', label = 'Russia')
sns.pointplot(data = India, x = India.index, y= 'confirmed', color = 'orange', label = 'India')


plt.xlabel('No of days', fontsize = 5)
plt.ylabel('Confirmed Cases', fontsize = 5)
plt.title('Confirmed cases over the period of time for top countries', fontsize = 5)
plt.xticks(rotation = 90, fontsize = 2)
plt.legend()
plt.show()


# In[50]:


sns.pointplot(data = Brazil, x = Brazil.index, y= 'active', color = 'green', label = 'Brazil')
sns.pointplot(data = US, x = US.index, y= 'active', color = 'red', label = 'US')
sns.pointplot(data = Russia, x = Russia.index, y= 'active', color = 'blue', label = 'Russia')
sns.pointplot(data = India, x = India.index, y= 'active', color = 'orange', label = 'India')


plt.xlabel('No of days', fontsize = 5)
plt.ylabel('Confirmed Cases', fontsize = 5)
plt.title('Confirmed cases over the period of time for top countries', fontsize = 5)
plt.xticks(rotation = 90, fontsize = 2)
plt.legend()
plt.show()


# In[51]:


sns.pointplot(data = Brazil, x = Brazil.index, y= 'recovered', color = 'green', label = 'Brazil')
sns.pointplot(data = US, x = US.index, y= 'recovered', color = 'red', label = 'US')
sns.pointplot(data = Russia, x = Russia.index, y= 'recovered', color = 'blue', label = 'Russia')
sns.pointplot(data = India, x = India.index, y= 'recovered', color = 'orange', label = 'India')


plt.xlabel('No of days', fontsize = 5)
plt.ylabel('Confirmed Cases', fontsize = 5)
plt.title('Confirmed cases over the period of time for top countries', fontsize = 5)
plt.xticks(rotation = 90, fontsize = 2)
plt.legend()
plt.show()


# In[52]:


sns.pointplot(data = Brazil, x = Brazil.index, y= 'deaths', color = 'green', label = 'Brazil')
sns.pointplot(data = US, x = US.index, y= 'deaths', color = 'red', label = 'US')
sns.pointplot(data = Russia, x = Russia.index, y= 'deaths', color = 'blue', label = 'Russia')
sns.pointplot(data = India, x = India.index, y= 'deaths', color = 'orange', label = 'India')


plt.xlabel('No of days', fontsize = 5)
plt.ylabel('Confirmed Cases', fontsize = 5)
plt.title('Confirmed cases over the period of time for top countries', fontsize = 5)
plt.xticks(rotation = 90, fontsize = 2)
plt.legend()
plt.show()


# In[53]:


get_ipython().system('pip install prophet')


# In[54]:


from prophet import Prophet


# In[55]:


df.head()


# In[56]:


df.groupby(by='date').sum().head()


# In[57]:


total_active = df['active'].sum()
print('Total number of active cases around the world: ', total_active)


# In[68]:


confirmed = df.groupby('date').sum()['confirmed'].reset_index()
death = df.groupby('date').sum()['deaths'].reset_index()
recovered = df.groupby('date').sum()['recovered'].reset_index()
active = df.groupby('date').sum()['active'].reset_index()


# In[59]:


confirmed.columns = ['ds', 'y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.tail()


# In[60]:


m = Prophet(interval_width = 0.95)
m.fit(confirmed)


# In[61]:


future = m.make_future_dataframe(periods = 7, freq = 'D')
future


# In[62]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)


# In[63]:


confirmed_forecast_plot = m.plot(forecast)


# In[64]:


death.columns = ['ds', 'y']
death['ds'] = pd.to_datetime(death['ds'])
m = Prophet()
m.fit(death)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
death_forecast_plot = m.plot(forecast)


# In[69]:


recovered.columns = ['ds', 'y']
recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet()
m.fit(recovered)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
recovered_forecast_plot = m.plot(forecast)


# In[70]:


active.columns = ['ds', 'y']
active['ds'] = pd.to_datetime(active['ds'])
m = Prophet()
m.fit(active)
future = m.make_future_dataframe(periods= 7, freq = "D")
forecast = m.predict(future)
active_forecast_plot = m.plot(forecast)


# In[ ]:




