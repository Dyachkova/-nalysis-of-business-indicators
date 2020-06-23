#!/usr/bin/env python
# coding: utf-8

# ## Аналитика в Яндекс.Афише

# #### Описание проекта
# Вас пригласили на стажировку в отдел аналитики Яндекс.Афиши. 
# - Первое задание: помочь маркетологам оптимизировать маркетинговые затраты.
# 
# У вас в распоряжении есть данные от Яндекс.Афиши с июня 2017 по конец мая 2018 года:
# - лог сервера с данными о посещениях сайта Яндекс.Афиши,
# - выгрузка всех заказов за этот период,
# - статистика рекламных расходов.
# 
# Вы изучите:
# - как люди пользуются продуктом,
# - когда они начинают покупать,
# - сколько денег приносит каждый клиент
# - когда клиент окупается.

# #### Шаг 1. Загрузите данные и подготовьте их к анализу
# - Загрузите данные о визитах, заказах и расходах в переменные.
# - Оптимизируйте данные для анализа.
# - Убедитесь, что тип данных в каждой колонке

# In[1]:


#Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns


# In[2]:


visits = pd.read_csv('/datasets/visits_log.csv', sep = ',')
visits.info()
visits.head()


# In[3]:


#заменим названия столбцов (приведем к нижнему регистру)
visits.columns = visits.columns.str.lower()
#переименуем колонки
visits.rename(columns={'end ts': 'end_ts', 'start ts': 'start_ts', 'source id': 'source_id'}, inplace=True)


# In[4]:


#изменим формат данных в колонках начала и окончания сесиии
visits['start_ts'] = pd.to_datetime(visits['start_ts'], format="%Y-%m-%d %H:%M")
visits['end_ts'] = pd.to_datetime(visits['end_ts'], format="%Y-%m-%d %H:%M")
visits.info()
visits.head()


# In[5]:


visits['device'].unique()


# In[6]:


orders = pd.read_csv('/datasets/orders_log.csv', sep = ',')
orders.info()
orders.head()


# In[7]:


#заменим названия столбцов (приведем к нижнему регистру)
orders.columns = orders.columns.str.lower()
#переименуем колонки
orders.rename(columns={'buy ts': 'buy_ts'}, inplace=True)
#изменим тип данных в колонке buy_ts
orders['buy_ts'] = pd.to_datetime(orders['buy_ts'], format="%Y-%m-%d %H:%M")
orders.info()


# In[8]:


costs = pd.read_csv('/datasets/costs.csv', sep = ',')
costs.info()
costs.head()


# In[9]:


#изменим формат данных в колонке с датой
costs['dt'] = pd.to_datetime(costs['dt'], format="%Y-%m-%d %H:%M")
costs.info()


# ### Вывод
# 

# Для анализа представлено три набора данных: visits, orders, costs. Проведена предобработка: названия столбцов приведены к нижнему регистру, переименованы, тип данных в необходимых столбцах изменен на datetime. Данные готовы к последующему анализу.

# #### Шаг 2. Постройте отчёты и посчитайте метрики
# Продукт
# Сколько людей пользуются в день, неделю, месяц? Сколько сессий в день? Сколько длится одна сессия? Как часто люди возвращаются?
# 
# Продажи
# Когда люди начинают покупать? Сколько раз покупают за период? Какой средний чек? Сколько денег приносят? (LTV)
# 
# Маркетинг
# Сколько денег потратили? Всего / на каждый источник / по времени Сколько стоило привлечение одного покупателя из каждого источника? На сколько окупились расходы? (ROI)
# 
# Отобразите на графиках, как эти метрики отличаются по устройствам и по рекламным источникам? Как они меняются во времени?

# In[10]:


#ПРОДУКТ

#добавим в таблицу visits колонки с указанием недели, месяца и года
visits['session_week'] = visits['start_ts'].dt.week
visits['session_month'] = visits['start_ts'].dt.month
visits['session_year'] = visits['start_ts'].dt.year
visits['session_date'] = visits['start_ts'].dt.date
visits.head(10)


# In[11]:


#Сколько людей пользуются в день, неделю, месяц?

dau_total = visits.groupby('session_date').agg({'uid': 'nunique'}).mean()
wau_total = visits.groupby(['session_year', 'session_week']).agg({'uid': 'nunique'}).mean()
mau_total = visits.groupby(['session_year', 'session_month']).agg({'uid': 'nunique'}).mean()
print('среднее количество уникальных пользователей в день:',int(dau_total))
print('среднее количество уникальных пользователей в неделю:',int(wau_total))
print('среднее количество уникальных пользователей в месяц:',int(mau_total))


# In[12]:


#построим графики по метрике DAU (Dayly Active Users).
#первый график - общий, второй - с разбивкой по устройствам, третий - с разбивкой по источнику
plt.subplots(figsize = (15,5))
visits.groupby(['session_date']).agg({'uid': 'nunique'}).plot(ax = plt.subplot(1,3,1))
plt.xticks(rotation=45)
(visits.groupby(['session_date','device'])
      .agg({'uid': 'nunique'})
      .pivot_table(index = 'session_date', columns = 'device', values = 'uid')
      .plot(ax = plt.subplot(1,3,2)))
plt.xticks(rotation=45)
(visits.groupby(['session_date','source_id'])
      .agg({'uid': 'nunique'})
      .pivot_table(index = 'session_date', columns = 'source_id', values = 'uid')
      .plot(ax = plt.subplot(1,3,3)))
plt.xticks(rotation=45)
plt.show()


# In[13]:


#построим графики по метрике WAU (Weekly Active Users)
#первый график - общий, второй - с разбивкой по устройствам, третий - с разбивкой по источнику
plt.subplots(figsize = (15,5))
visits.groupby(['session_year', 'session_week']).agg({'uid': 'nunique'}).plot(ax = plt.subplot(1,3,1))
plt.xticks(rotation=45)
(visits.groupby(['session_year', 'session_week','device'])
      .agg({'uid': 'nunique'})
      .pivot_table( index = ['session_year', 'session_week'], columns = 'device', values = 'uid')
      .plot(ax = plt.subplot(1,3,2)))
plt.xticks(rotation=45)
(visits.groupby(['session_year', 'session_week','source_id'])
      .agg({'uid': 'nunique'})
      .pivot_table( index = ['session_year', 'session_week'], columns = 'source_id', values = 'uid')
      .plot(ax = plt.subplot(1,3,3)))
plt.xticks(rotation=45)
plt.show()


# In[14]:


#построим графики по метрике MAU (Monthly Active Users)
#первый график - общий, второй - с разбивкой по устройствам, третий - с разбивкой по источнику
plt.subplots(figsize = (15,5))
visits.groupby(['session_year', 'session_month']).agg({'uid': 'nunique'}).plot(ax = plt.subplot(1,3,1))
plt.xticks(rotation=45)
(visits.groupby(['session_year', 'session_month','device'])
      .agg({'uid': 'nunique'})
      .pivot_table( index = ['session_year', 'session_month'], columns = 'device', values = 'uid')
      .plot(ax = plt.subplot(1,3,2)))
plt.xticks(rotation=45)
(visits.groupby(['session_year', 'session_month','source_id'])
      .agg({'uid': 'nunique'})
      .pivot_table( index = ['session_year', 'session_month'], columns = 'source_id', values = 'uid')
      .plot(ax = plt.subplot(1,3,3)))
plt.xticks(rotation=45)
plt.show()


# Пользователи заходят с desktop версии в основном
# По источникам наибольшее количество пользователей пришло с источников: 4,3,5

# In[15]:


#Сколько сессий в день?

visits.groupby('session_date').agg({'uid': 'count'}).plot()
plt.xticks(rotation=45)
plt.show() 


# In[16]:


sessions = visits.groupby('session_date').agg({'uid': 'count'})
sessions.describe()


# In[17]:


#Сколько длится одна сессия? 
visits['session_duration_sec'] = (visits['end_ts'] - visits['start_ts']).dt.total_seconds()
visits.boxplot('session_duration_sec', by = 'device', figsize = (10, 5))
plt.ylim (0, 3000)
plt.show()


# In[18]:


#Сколько длится одна сессия? 
visits['session_duration_sec'] = (visits['end_ts'] - visits['start_ts']).dt.total_seconds()
visits['session_duration_sec'].describe()


# Количество сессий в большинстве случаев равно 1003 в день, длительность сессии составляет 300 сек.

# In[19]:


#сделаем срез по количеству сессий на каждой из платформ
session_by_device = visits.groupby(['session_year', 'session_month', 'device'])                          .agg({'uid': 'nunique'})
session_by_device


# In[20]:


#Как часто люди возвращаются?
#sticky factor = DAU/MAU

sticky_factor = round((dau_total / mau_total)*100)
print('повторные визиты пользователей в течение месяца:', sticky_factor)


# In[21]:


#добавим в таблицу дату первой активности
#удалим в датафрейме visits ненужные нам столбцы
visits = visits.drop(['end_ts', 'session_week', 'session_month', 'session_year', 'session_duration_sec'], axis=1)
first_session_date = visits.groupby(['uid'])['session_date'].min()
first_session_date.name = 'first_session_date'
visits = visits.merge(first_session_date, on='uid')
visits.head()


# In[22]:


#выделим месяц первой активности
visits['first_session_month'] = visits['first_session_date'].astype('datetime64[M]')
visits['last_session_month'] = visits['session_date'].astype('datetime64[M]')

visits.head()


# In[23]:


#найдем лайфтайм когорты
visits['cohort_lifetime'] = visits['last_session_month'] - visits['first_session_month']
visits['cohort_lifetime'] = visits['cohort_lifetime'] / np.timedelta64(1,'M')
visits['cohort_lifetime'] = visits['cohort_lifetime'].round().astype(int)


# In[24]:


#Посчитаем для каждой когорты количество активных пользователей на определённый «месяц жизни»:
cohorts = visits.groupby(['first_session_month','cohort_lifetime']).agg({'uid':'nunique'}).reset_index()
cohorts.head(10)


# In[25]:


#Найдём исходное количество пользователей в когорте. Возьмём их число на нулевой месяц:
initial_users_count = cohorts[cohorts['cohort_lifetime'] == 0][['first_session_month','uid']]
initial_users_count = initial_users_count.rename(columns={'uid':'cohort_users'})
#Объединим данные по когортам с исходным количеством пользователей в когорте:
cohorts = cohorts.merge(initial_users_count,on='first_session_month')
print(initial_users_count)


# Активность начинает расти с сентября. 6 месяцев -с октября по март- значительная активность (концентрация на внутригородских развлечениях и три больших праздника - Новый год, 23 февраля, 8 марта). 

# In[26]:


#рассчитаем Retention Rate
#Разделим количество активных пользователей в каждом месяце на исходное число пользователей в когорте:
cohorts['retention'] = cohorts['uid']/cohorts['cohort_users']


# In[27]:


#Построим сводную таблицу и создадим тепловую карту:
retention_pivot = cohorts.pivot_table(index='first_session_month',columns='cohort_lifetime',values='retention',aggfunc='sum')
retention_pivot


# In[28]:


sns.set(style='white')
plt.figure(figsize=(13, 9))
plt.title('Cohorts: User Retention')
sns.heatmap(retention_pivot, annot=True, fmt='.1%', linewidths=1, linecolor='gray')


# Retention rate постоянно снижается - это видно с каждой когортой. Надо больше работать над удержанием клиентов.

# In[29]:


#добавим в таблицу orders колонки с указанием дня и месяца

orders['orders_month'] = orders['buy_ts'].dt.month
orders['orders_year'] = orders['buy_ts'].dt.year
orders['orders_date'] = orders['buy_ts'].dt.date
orders.head(5)


# In[30]:


#Когда люди начинают покупать?

#найдем сколько времени проходит от начала первой зарегистрированной сессии до первой покупки
first_activity = visits.groupby(['uid'])['start_ts'].min().reset_index()
first_buy = orders.groupby(['uid'])['buy_ts'].min().reset_index()
orders_new = (first_buy.merge(first_activity, on = 'uid')
                .rename(columns = {'start_ts': 'first_activity', 'buy_ts': 'first_buy'}))

orders_new['time_to_buy'] = ((orders_new['first_buy'] - orders_new['first_activity']).dt.total_seconds())/60

print('среднее время до покупки:', int(orders_new['time_to_buy'].median()),'минут')


# In[31]:


#сумма и количество покупок в месяц:

plt.subplots(figsize = (20,5))
orders.groupby(['orders_year', 'orders_month']).agg({'revenue': 'sum'}).plot(ax = plt.subplot(1,2,1))

orders.groupby(['orders_year', 'orders_month']).agg({'revenue': 'count'}).plot(ax = plt.subplot(1,2,2))

plt.show()
#как и количество пользователей, покупки значительно растут с октября до Нового года, затем поддерживается высокий 
#уровень до марта, с апреля уровень покупок снижается значительно 


# In[32]:


#сумма и количество покупок в день:

plt.subplots(figsize = (20,5))
orders.groupby('orders_date').agg({'revenue': 'sum'}).plot(ax = plt.subplot(1,2,1))
plt.xticks(rotation=45)
orders.groupby('orders_date').agg({'revenue': 'count'}).plot(ax = plt.subplot(1,2,2))
plt.xticks(rotation=45)
plt.show()


# In[33]:


#Какой средний чек?
#Для каждого клиента получим дату его первого заказа
first_order_date_by_customers = orders.groupby('uid')['buy_ts'].min() 
first_order_date_by_customers.name = 'first_order_date'
orders = orders.join(first_order_date_by_customers,on='uid')
orders.head(5)


# In[34]:


#выделим месяцы из столбцов buy_ts и first_order_date
orders['first_order_month'] = orders['first_order_date'].astype('datetime64[M]')
orders['order_month'] = orders['buy_ts'].astype('datetime64[M]')


# In[35]:


#Когортой станет столбец first_order_month — месяц, в котором был сделан первый заказ. 
#Сгруппируем данные по этому столбцу и оценим показатели каждой когорты.
cohort_grouped = orders.groupby('first_order_month').agg({'uid':'nunique','revenue':'sum'})
print(cohort_grouped)


# In[36]:


orders_grouped_by_cohorts = orders.groupby(['first_order_month','order_month']).agg({'revenue':'sum','uid':'nunique'})
orders_grouped_by_cohorts.head(10)


# In[37]:


#найдем средний чек покупателя
orders_grouped_by_cohorts['revenue_per_user'] = orders_grouped_by_cohorts['revenue'] / orders_grouped_by_cohorts['uid']
orders_grouped_by_cohorts


# In[38]:


#найдем средний чек покупателя
orders_grouped_by_cohorts.pivot_table(index = 'order_month', values = 'revenue_per_user', aggfunc = 'sum')
#Средний чек растет с начала "активного сезона" - в сентябре, значительно увеличивается в декабре (в три раза), 
#держится на уровне в 2 раза больше в период февраль - май


# In[39]:


#Построим сводную таблицу изменения среднего чека в когортах по месяцу совершения покупки и оценим, 
#как изменяется средний чек с течением времени:
orders_grouped_by_cohorts.pivot_table(index='first_order_month',columns='order_month',values='revenue_per_user',aggfunc='mean')
#средний чек стабильно увеличивается на 2й месяц, пик затрат будет в периоде 3-7 месяца


# #### Маркетинг
# - Сколько денег потратили? Всего / на каждый источник / по времени
# - Сколько стоило привлечение одного покупателя из каждого источника?
# - На сколько окупились расходы? (ROI)

# In[40]:


#Сколько денег потратили? Всего / на каждый источник / по времени
costs['month'] = costs['dt'].astype('datetime64[M]')
costs.pivot_table(index = ['month'], columns = 'source_id', values = 'costs')


# In[41]:


# сколько потратили на маркетинг по рекламным источникам:
costs


# In[42]:


# сколько потратили на маркетинг по рекламным источникам:
costs.groupby(['source_id']).agg({'costs': 'sum'}).plot(figsize = (10,5))
plt.show()
#больше всего потратили на 3 рекламный источник


# In[43]:


# сколько потратили на маркетинг по рекламным источникам, в месяц:
costs.pivot_table(index = ['month'], columns = 'source_id', values = 'costs').round(2).plot(figsize = (10,5))


# In[44]:


# сколько потратили на маркетинг по месяцам, всего:
costs_total = costs.groupby(['month']).agg({'costs': 'sum'}).plot(figsize = (10,5))
plt.show()


# CAC (от англ. customer acquisition cost) — стоимость привлечения клиента. 
# Сумма денег, во сколько обходится новый клиент компании.
# Экономика одного покупателя сходится, если LTV больше CAC.
# 

# In[45]:


#Сколько стоило привлечение одного покупателя из каждого источника?(САС)

# считаем расходы за месяц
monthly_costs = costs.groupby(['month','source_id']).agg({'costs': 'sum'})
monthly_costs


# In[46]:


#Для каждого пользователя определим из visits дату и источник первого посещения
first_visits_source_id = visits.query('session_date == first_session_date')
first_visits_source_id = first_visits_source_id.drop(['device','start_ts',
                                                      'session_date','first_session_month',
                                                      'last_session_month','cohort_lifetime'], axis=1)
                         
first_visits_source_id.head(5)


# In[47]:


first_visits_source_id['first_session_date'] = pd.to_datetime(first_visits_source_id['first_session_date'], format="%Y-%m-%d %H:%M")
first_visits_source_id.info()


# In[48]:


costs.head()


# In[49]:


#строим профили юзеров
users = (visits.sort_values(by = ['uid', 'start_ts'])
               .groupby('uid')
               .agg({'start_ts': 'first', 'source_id': 'first', 'device': 'first'})
               .rename(columns = {'start_ts':  'acquisition_ts'}))

users['acquisition_date'] = users['acquisition_ts'].dt.date
users['acquisition_month'] = users['acquisition_ts'].astype('datetime64[M]')    
costs.rename(columns={'dt': 'acquisition_date'}, inplace=True)
users = (users.reset_index()
              .set_index(['source_id', 'acquisition_date'])
              .join(costs.set_index(['source_id', 'acquisition_date']), how = 'left'))


# In[50]:


#добавляем стоимость приобретения индивидуального пользователя
user_cost = (users.groupby(['source_id', 'acquisition_date'])
                  .agg({'device': 'count', 'costs': 'max'})
                  .rename(columns = {'device': 'users'})) 
users.head()


# In[51]:


#посчитаем стоимость привлечения одного пользователя:

user_cost['acquisition_cost'] = user_cost['costs'] / user_cost['users']
users = users.join(user_cost['acquisition_cost'], how = 'left').reset_index()

users = users.set_index('uid')[['acquisition_ts', 'acquisition_date', 'acquisition_month',
                               'source_id', 'device', 'acquisition_cost']]
#платящий пользователь или нет
users = users.join(orders.groupby('uid').agg({'buy_ts': 'min'}).rename(columns = {'buy_ts': 'first_purchase_dt'}), how = 'left')
users['payer'] = ~users['first_purchase_dt'].isna()


# In[52]:


users


# In[53]:


#посчитаем когорты и CAC(стоимость привлечения клиента)
cohorts_new = (users.groupby('acquisition_month')
                    .agg({'payer': 'sum', 'acquisition_cost': 'sum'})
                    .rename(columns = {'payer': 'cohort_size'}))

cohorts_new['cac'] = cohorts_new['acquisition_cost'] / cohorts_new['cohort_size']


# In[54]:


#считаем LTV
ltv = orders.set_index('uid').join(users, how = 'left')[['acquisition_month', 'first_order_month', 'revenue']]
ltv['age_month'] = ((ltv['first_order_month'] - ltv['acquisition_month']) / np.timedelta64(1, 'M')).round().astype('int')
ltv = ltv.groupby(['acquisition_month', 'age_month']).agg({'revenue': 'sum'})
ltv = pd.pivot_table(ltv, index = 'acquisition_month', columns = 'age_month', values = 'revenue', aggfunc = 'sum')
ltv = ltv.cumsum(axis = 1)
ltv = cohorts_new[['cohort_size']].join(ltv)
ltv = ltv.div(ltv['cohort_size'], axis = 0).drop(columns = ['cohort_size']) #делим все ячейки в рядах на соответствующую когорту
cohorts_new[['cohort_size', 'cac']].join(ltv[range(0, 6)].fillna(''))


# In[55]:


#LTV с разбивкой по УСТРОЙСТВУ

#посчитаем когорты и CAC(стоимость привлечения клиента)
cohorts_new = (users.groupby('device')
                    .agg({'payer': 'sum', 'acquisition_cost': 'sum'})
                    .rename(columns = {'payer': 'cohort_size'}))

cohorts_new['cac'] = cohorts_new['acquisition_cost'] / cohorts_new['cohort_size']

#считаем LTV
ltv = orders.set_index('uid').join(users, how = 'left')[['acquisition_month', 'first_order_month', 'revenue', 'device']]
ltv['age_month'] = ((ltv['first_order_month'] - ltv['acquisition_month']) / np.timedelta64(1, 'M')).round().astype('int')
ltv = ltv.groupby(['device', 'age_month']).agg({'revenue': 'sum'})
ltv = pd.pivot_table(ltv, index = 'device', columns = 'age_month', values = 'revenue', aggfunc = 'sum')
ltv = ltv.cumsum(axis = 1)
ltv = cohorts_new[['cohort_size']].join(ltv)
ltv = ltv.div(ltv['cohort_size'], axis = 0).drop(columns = ['cohort_size']) #делим все ячейки в рядах на соответствующую когорту
cohorts_new[['cohort_size', 'cac']].join(ltv[range(0, 6)].fillna(''))


# In[56]:


#LTV с разбивкой по рекламному ИСТОЧНИКУ

#посчитаем когорты и CAC(стоимость привлечения клиента)
cohorts_new = (users.groupby('source_id')
                    .agg({'payer': 'sum', 'acquisition_cost': 'sum'})
                    .rename(columns = {'payer': 'cohort_size'}))

cohorts_new['cac'] = cohorts_new['acquisition_cost'] / cohorts_new['cohort_size']

#считаем LTV
ltv = orders.set_index('uid').join(users, how = 'left')[['acquisition_month', 'first_order_month', 'revenue', 'source_id']]
ltv['age_month'] = ((ltv['first_order_month'] - ltv['acquisition_month']) / np.timedelta64(1, 'M')).round().astype('int')
ltv = ltv.groupby(['source_id', 'age_month']).agg({'revenue': 'sum'})
ltv = pd.pivot_table(ltv, index = 'source_id', columns = 'age_month', values = 'revenue', aggfunc = 'sum')
ltv = ltv.cumsum(axis = 1)
ltv = cohorts_new[['cohort_size']].join(ltv)
ltv = ltv.div(ltv['cohort_size'], axis = 0).drop(columns = ['cohort_size']) #делим все ячейки в рядах на соответствующую когорту
ltv_source = cohorts_new[['cohort_size', 'cac']].join(ltv[range(0, 6)].fillna(''))


# In[57]:


ltv_source


# уберем из расчетов 6 и 7 источники, по ним недостаточно данных для анализа

# In[58]:


ltv_source = ltv_source.query('source_id != "6" and source_id != "7"')
ltv_source


# In[59]:


#Рассчитаем ROMI: поделим LTV на CAC.
roi = ltv_source.div(ltv_source['cac'], axis = 0).drop(columns = ['cohort_size', 'cac']).fillna('')
roi


# Наиболее выгодные источники для вложений: 1, 2, 5 и 9

# ### Вывод

# среднее количество уникальных пользователей в день: 907
# среднее количество уникальных пользователей в неделю: 5716
# среднее количество уникальных пользователей в месяц: 23228
# 
# Пользователи заходят с desktop версии в основном
# По источникам наибольшее количество пользователей пришло с источников: 4,3,5
# 
# среднее количество сессий - 987 в день
# Увеличение идет с октября: люди вернулись с отпусков и с дач. Пик перед новогодними праздниками - покупка подарков. Следующие два пика, меньше - перед праздниками 23 февраля и 8 марта.  Сразу после 8 марта резкое снижение: билеты уже куплены, деньги на праздники потрачены. Общее постепенное снижение к лету: люди переключаются с внутригородских развлечений на поездки за город и отпуск.
# 
# 
# 
# повторных визитов пользователей в течение месяца - 4
# среднее время до покупки: 16 минут
# 
# Средний чек на второй месяц стабильно увеличивается.
# Средний чек растет с начала "активного сезона" - в сентябре, значительно увеличивается в декабре (в три раза), 
# держится на уровне в 2 раза больше в период февраль - май
# наиболее выгодная когорта - сентябрьская, вложения на ее привлечение окупаются быстрее всего
# 
# В среднем когорты окупаются на 8 месяц

# Шаг 3. Напишите вывод: порекомендуйте маркетологам, куда и сколько им стоит вкладывать денег?
# Какие источники/платформы вы бы порекомендовали? Объясните свой выбор: на какие метрики вы ориентируетесь? Почему? Какие выводы вы сделали, узнав значение метрик?

# 
# Коэффициент удержания снижается с каждой когортой.
# 
# - для постоянных покупателей
# - сервис анализа предпочтений (наподобие Яндекс.Музыки) - советы по похожим мероприятиям
# - трекинг мероприятий по любимым исполнителям/организаторам с настроенными уведомлениями
# 
# Количество сессий в большинстве случаев равно 1003 в день, длительность сессии составляет 300 сек.
# 
# Наиболее выгодные источники для вложений: 1, 2, 5 и 9
# Несмотря на то, что основные затраты приходятся на 3 и 4 источники, прибыль они приносят очень неравномерно. Источник 3 окупается меньше, чем наполовину.
