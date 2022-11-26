import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_regression
import re
#数据读入(时间列*变量行)
dt=pd.read_csv(r'C:\Users\lhm20\data_all.csv')
dt

##更改列名
#df.rename(columns={'旧列名':'新列名'})
dt.rename(columns={'Unnamed: 0':'date'})
testpi=pd.DataFrame(None ,index =dt.iloc[:,0],columns=["0","1","2","3","4","5","6","7","8","9","10","11","12"])
testimportance=pd.DataFrame(None ,index =dt.iloc[4:128,0],columns=["feature","0","1","2","3","4","5","6","7","8","9","10","11","12"])
testfeatures=pd.DataFrame(None ,index =dt.iloc[0:20,0],columns=["0","1","2","3","4","5","6","7","8","9","10","11","12"])
begin_date='1/1/1990'#360
end_date='12/1/2021'#744

####滚动回归
    ###开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
time=dt.iloc[:,0]
    #检查
time
    #取第一列变量做成列表
col=time.tolist()

for h in range(13):
        #限定训练集数据的范围
        #样本内窗口初始为60-89，一共30年，设为train_range
    for i in range(col.index(end_date)-col.index(begin_date)+1):
        train_range=360
        train_X=dt.iloc[i:train_range+i-1-h,2:]
        train_y=dt.iloc[i:train_range+i-1-h,1]
        train_y=pd.DataFrame(train_y)
        test_X=dt.iloc[train_range+i,2:]
        test_X=pd.DataFrame(test_X)
        test_X=test_X.T
        test_y=dt.iloc[train_range+i,1]
        rf_train=ExtraTreesRegressor()
        rf_train.fit(train_X,train_y)
        test_pi=rf_train.predict(test_X)
        len(test_pi)
        test_pi=pd.DataFrame(test_pi)
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        testimportance.iloc[:,0]=features
        testimportance.iloc[:,h+1]=importance
        n_num=20
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        test1=model.get_support(indices=True)
        testfeatures.iloc[0:20,h]=test1
        testpi.iloc[i,h]=test_pi.iloc[0,0]
    print(h)
testpi.to_csv(r'C:\Users\lhm20\testpi.csv')
testimportance.to_csv(r'C:\Users\lhm20\testimportance.csv')
testfeatures.to_csv(r'C:\Users\lhm20\testfeatures.csv')

##进一步考虑到累计值为3，6，12预测，增加变量 ver3,ver6,ver12
#数据读入(时间列*变量行)
dt=pd.read_csv(r'C:\Users\lhm20\data_all.csv')
dt
##更改列名
#df.rename(columns={'旧列名':'新列名'})
dt.rename(columns={'Unnamed: 0':'date'})

##分别构建数据集，进行移动求和_注意再在csv里面改一下数据,上挪2年数据（要保证样本外窗口数据是完整的)
#三年相加
for i in range(744):
    ver3=dt.rolling(window=3).sum()
    ver3
test=pd.DataFrame(ver3)
test.to_csv(r'C:\Users\lhm20\1960-2021-ver3.csv',encoding = 'gbk')

##分别构建数据集，进行移动求和_注意再在csv里面改一下数据,上挪5年数据（要保证样本外窗口数据是完整的)
#六年相加
for i in range(744):
    ver6=dt.rolling(window=6).sum()
    ver6
test =pd.DataFrame(ver6)
test.to_csv(r'C:\Users\lhm20\1960-2021-ver6.csv',encoding = 'gbk')

##分别构建数据集，进行移动求和_注意再在csv里面改一下数据,上挪11年数据（要保证样本外窗口数据是完整的)
#十二年相加
for i in range(744):
    ver12=dt.rolling(window=12).sum()
    ver12
test =pd.DataFrame(ver12)
test.to_csv(r'C:\Users\lhm20\1960-2021-ver12.csv',encoding = 'gbk')    

##acc=3
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

data3=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver3.csv')
data3

##更改列名
#df.rename(columns={'旧列名':'新列名'})
data3=data3.rename(columns={'Unnamed: 0':'date'})

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
data3.dropna(subset=['pi'],inplace=True,axis=0)

def rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=358
        #取训练集的数据
        train_X=dt.iloc[train_range-358+i:train_range+i,2:]
        train_y=dt.iloc[train_range-358+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:742,2:]
        test_y=dt.iloc[train_range+i+1:742,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
        #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\rollingrf-acc3-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\rollingRF-acc3-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
      
##调用函数
begin_date='1/1/1990'
end_date='12/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=rolling_rf(data3,b_d,e_d)
##看一下自己的数据什么情况
pre

#没有问题的话，进行输出
test = pd.DataFrame(pre)
test.to_csv(r'C:\Users\lhm20\testpi-ver3.csv',encoding = 'gbk')

##acc=6
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

data6=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver6.csv')
data6

##更改列名
#df.rename(columns={'旧列名':'新列名'})
data6=data6.rename(columns={'Unnamed: 0':'date'})
data6

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
data6.dropna(subset=['pi'],inplace=True,axis=0)

def rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=355
        #取训练集的数据
        train_X=dt.iloc[train_range-355+i:train_range+i,2:]
        train_y=dt.iloc[train_range-355+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:739,2:]
        test_y=dt.iloc[train_range+i+1:739,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
        #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\rollingrf-acc6-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\rollingRF-acc6-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
      
##调用函数
begin_date='1/1/1990'
end_date='12/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=rolling_rf(data6,b_d,e_d)
##看一下自己的数据什么情况
pre

##acc=12
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

data12=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver12.csv')
data12

##更改列名
#df.rename(columns={'旧列名':'新列名'})
data12=data12.rename(columns={'Unnamed: 0':'date'})
data12

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
data12.dropna(subset=['pi'],inplace=True,axis=0)
data12

def rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=349
        #取训练集的数据
        train_X=dt.iloc[train_range-349+i:train_range+i,2:]
        train_y=dt.iloc[train_range-349+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:733,2:]
        test_y=dt.iloc[train_range+i+1:733,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
        #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\rollingrf-acc12-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\rollingRF-acc12-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
    
##调用函数
begin_date='1/1/1990'
end_date='12/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=rolling_rf(data12,b_d,e_d)
##看一下自己的数据什么情况
pre

#没有问题的话，进行输出
test = pd.DataFrame(pre)
test.to_csv(r'C:\Users\lhm20\testpi-ver12.csv',encoding = 'gbk')

###考虑到中心值设定没有说明，因此我们对其进行了处理，选择训练集的数据完整，测试机数据减少

##acc=3
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

#由于在文件‘1960-2021-ver3.csv’生成后，手动修改上挪2行的文件未单独另存为，直接在原表修改，则此处重新导入一次数据
d_data3=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver3.csv')
d_data3

##更改列名
#df.rename(columns={'旧列名':'新列名'})
d_data3=d_data3.rename(columns={'Unnamed: 0':'date'})

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
d_data3.dropna(subset=['pi'],inplace=True,axis=0)
d_data3

def r_rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=360
        #取训练集的数据
        train_X=dt.iloc[train_range-360+i:train_range+i,2:]
        train_y=dt.iloc[train_range-360+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:742,2:]
        test_y=dt.iloc[train_range+i+1:742,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
        #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\r_rollingrf-acc3-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\r_rollingRF-acc3-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
    
##调用函数
begin_date='1/1/1990'
end_date='10/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=r_rolling_rf(d_data3,b_d,e_d)
##看一下自己的数据什么情况
pre

#没有问题的话，进行输出
test = pd.DataFrame(pre)
test.to_csv(r'C:\Users\lhm20\t_testpi-ver3.csv',encoding = 'gbk')

##acc=6
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

#由于在文件‘1960-2021-ver6.csv’生成后，手动修改上挪5行的文件未单独另存为，直接在原表修改，则此处重新导入一次数据
d_data6=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver6.csv')
d_data6

##更改列名
#df.rename(columns={'旧列名':'新列名'})
d_data6=d_data6.rename(columns={'Unnamed: 0':'date'})

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
d_data6.dropna(subset=['pi'],inplace=True,axis=0)
d_data6

def r_rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=360
        #取训练集的数据
        train_X=dt.iloc[train_range-360+i:train_range+i,2:]
        train_y=dt.iloc[train_range-360+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:739,2:]
        test_y=dt.iloc[train_range+i+1:739,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
        #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\r_rollingrf-acc6-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\r_rollingRF-acc6-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
    
##调用函数
begin_date='1/1/1990'
end_date='7/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=r_rolling_rf(d_data6,b_d,e_d)
##看一下自己的数据什么情况
pre

#没有问题的话，进行输出
test = pd.DataFrame(pre)
test.to_csv(r'C:\Users\lhm20\t_testpi-ver6.csv',encoding = 'gbk')

##acc=12
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

#由于在文件‘1960-2021-ver12.csv’生成后，手动修改上挪11行的文件未单独另存为，直接在原表修改，则此处重新导入一次数据
d_data12=pd.read_csv(r'C:\Users\lhm20\1960-2021-ver12.csv')
d_data12

##更改列名
#df.rename(columns={'旧列名':'新列名'})
d_data12=d_data12.rename(columns={'Unnamed: 0':'date'})

##删除行：drop(),axis=1删除列，axis=0删除行
#各种删除的函数可以参考：https://mp.weixin.qq.com/s?src=11&timestamp=1638672957&ver=3477&signature=rxO*QMcjp6dUXdIJ*qYxw*qpo77QtsOSf3e1I5AGjp7kcoFKm2pmd87*a-Nleo3*XPPWglMymSE2foKYTfi3zxSmJ7dyqCYr4-xuA6mXSKdG6c20tE9s-ff*bJOcq9xY&new=1
#此处删除由于移动求和导致的变量为空值的情况
d_data12.dropna(subset=['pi'],inplace=True,axis=0)
d_data12

def r_rolling_rf(dt,b_d,e_d):
    ####滚动回归
    #开始滚动，按照预测区间设计 range范围
    ####这里我们做了以下处理：
    #因为要取第一列的数，所以我们需要先取到时间这一列
    time=dt.iloc[:,0]
    #检查
    time
    #取第一列变量做成列表
    col=time.tolist()
    ##新循环函数代码设计
    ##问题1：一个是循环函数怎么输出没搞完
    ##问题2：还有一个是移动求和做累计值哪里，sum不知道该怎么对整体用，这里还没有做
    ##思路：目前是需要分成两个部分：第一部分是先嵌套上h，做滞后项，然后再嵌套回测函数；第二部分是先做累积值的三个表格，然后分别对他们进行滞后项为0的滚动回测。
    for i in range(e_d-b_d+1):
        #样本内窗口初始为60-89，一共30年，设为train_range
        train_range=360
        #取训练集的数据
        train_X=dt.iloc[train_range-360+i:train_range+i,2:]
        train_y=dt.iloc[train_range-360+i:train_range+i,1]
        #对测试集数据范围进行定义
        test_X=dt.iloc[train_range+i+1:733,2:]
        test_y=dt.iloc[train_range+i+1:733,1]
        #对训练集进行随机森林拟合处理
        #以及通过随机森林筛选最优变量
         #这里用极端随机森林
        rf_train=ExtraTreesRegressor()
        #这个原始用的是普通随机森林，普通随机森林模型构建的分裂结点是随机选取的特征，这里我们要用极端随机森林，也就是构建树的时候，不会任意选取特征，而是先随机收集一部分特征，然后利用信息熵、基尼指数挑选最佳结点特征。
        #普通随机：rf_train=RandomForestRegressor(max_depth=2, random_state=0)
        rf_train.fit(train_X,train_y)##regr.fit(data_feature,data_target)
         #进行变量重要性筛选，对变量贡献度进行打分
        importance = rf_train.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = train_X.columns
        ##导出打分结果
        test = pd.DataFrame(importance,features)
        test.to_csv(r'C:\Users\lhm20\r_rollingrf-acc12-importance.csv',encoding = 'gbk')
        #print得出，更加的直观一些
        #for f in range(train_X.shape[1]):
        #    print(("%2d) %-*s %f" % (f + 1, 30, features[f], importance[indices[f]])))
        ##选择前20个最优变量
        n_num=20##根据需要，选择要留几个特征值
        model=SelectFromModel(rf_train,prefit=True,max_features=n_num,threshold=-np.inf)
        dt_feature_new=pd.DataFrame(model.transform(train_X))
        #返回被筛选得到的特征值在文中的位置test1
        test1=model.get_support(indices=True)
        test = pd.DataFrame(dt_feature_new,test1)
        test.to_csv(r'C:\Users\lhm20\r_rollingRF-acc12-筛选变量.csv',encoding = 'gbk')
        #对训练集进行预测
        test_pi=rf_train.predict(test_X)
        return(test_pi)
    
##调用函数
begin_date='1/1/1990'
end_date='1/1/2021'
b_d=col.index(begin_date)
e_d=col.index(end_date)

pre=r_rolling_rf(d_data12,b_d,e_d)
##看一下自己的数据什么情况
pre

#没有问题的话，进行输出
test = pd.DataFrame(pre)
test.to_csv(r'C:\Users\lhm20\t_testpi-ver12.csv',encoding = 'gbk')  

##手动修改'testpi.csv'，另存为't_testpi.csv'(数据期间为1/1/1990-12/1/2021)
t_testpi=pd.read_csv(r'C:\Users\lhm20\t_testpi.csv')
t_testpi



