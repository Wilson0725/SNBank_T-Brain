# %% 
import pandas as pd
import numpy as py
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

# %% 
### find if nan

PATH = 'C:/Users/Wilson/Desktop/t-brain_dataset'
df = pd.read_csv(f'{PATH}/training_data.csv')

for i, col in enumerate(df.columns):
    if df[col].isna().any():
        print(i,col, df[col].dtype, df[col].isna().any())
    else:
        print('no nan')
        pass

# %%
### nan investigate 

remark = df["備註"]              # 11660 nan, 110 diff remark --> suggested to drop --> 手動處理 --> piority 4 
Counter(remark)

region = df["使用分區"]          # 11156 nan, 308 住, 86 商, 27 農, 25 工, 149 其他
Counter(region)

road_name = df["路名"]           # 
Counter(region)

# %% 
xy_dict = {}
x_coord = df['橫坐標']
y_coord = df['縱坐標']

city = df['縣市']

# %%
plt.scatter(x_coord,y_coord)
plt.show()


# %%
### def 

used_area = [
    '主建物面積',
    '車位面積',
    '附屬建物面積',
    '陽台面積'
    ]
city_price_dic = {}

def avg_price(df,city):
    target_df = df[df['縣市'] == city]
   
    area = 0 
    for a in used_area:
        area += target_df[a].sum() + (df[a].mean()*len(target_df[a])) + (df[a].mean()*len(target_df[a]))          # 全部加上一個 mean

    mean_price_per_area = (target_df['單價'].sum())/area

    return mean_price_per_area

for u in df['縣市'].unique():
    city_price_dic[f'{u}_nor'] = avg_price(df,u)

print(city_price_dic)
    
real_price_dic = {
    '台北市':78.0,
    '高雄市':23.5,
    '新北市':43.4,
    '桃園市':27.4,
    '台中市':28.8,
    '台南市':25.2,
    '苗栗縣':32.1,
    '新竹縣':35.0,
    '基隆市':22.2,
    '屏東縣':13.0,
    '新竹市':34.0,
    '宜蘭縣':29.9,
    '花蓮縣':26.6,
    '嘉義市':23.2,
    '金門縣':23.8,
    '嘉義縣':23.2,
    '彰化縣':27.9,
    '雲林縣':23.5
    }

# %%
### plot
plt.plot(real_price_dic.keys(),real_price_dic.values())
plt.plot(range(len(real_price_dic)),city_price_dic.values())
plt.xticks(range(len(real_price_dic)),rotation=90)

# %%
