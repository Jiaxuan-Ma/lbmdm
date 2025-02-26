'''
Runs the streamlit app
Call this file in the terminal via `streamlit run app.py`
'''


import shap

import streamlit as st

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from streamlit_shap import st_shap

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_validate as CV

from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import LeaveOneOut

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import shap
import matplotlib.pyplot as plt
import pickle
from utils import *
from streamlit_extras.badges import badge
import warnings

# import sys
from prettytable import PrettyTable

import scienceplots


warnings.filterwarnings('ignore')

st.set_page_config(
        page_title="DMMD",
        page_icon="🍁",
        layout="centered",
        initial_sidebar_state="auto",
        # menu_items={
        # })
        menu_items={
        })
sysmenu = '''
<style>
MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''

# https://icons.bootcss.com/
st.markdown(sysmenu,unsafe_allow_html=True)
# arrow-repeat
with st.sidebar:
    st.markdown(
    '''
    ```
    copyright

    MGI(上海大学材料基因组工程研究院)
    ```
            ''')
    select_option = option_menu("LBEDM", ["平台主页", "数据可视化", "特征工程", "回归预测", "可解释性机器学习","模型推理"],
                    icons=['house', 'clipboard-data', 'menu-button-wide','bezier2', 'arrow-repeat','subtract', 'app', 'microsoft'],
                    menu_icon="boxes", default_index=0)


df = pd.read_csv("./data/lbm.csv")
if select_option == "平台主页":
    st.write('''![](https://github.com/Jiaxuan-Ma/Transfer-Learning/assets/61132191/d2b0d25d-1353-46eb-8b8a-13f35dc0ebd5?raw=true)''')
    st.image("https://github.com/user-attachments/assets/18b314b2-f08c-454b-9dfd-3f0d6e32c2d0")
    colored_header(label="铅铋腐蚀数据挖掘平台",description="LBE data-mining",color_name="violet-90")


elif select_option == "数据可视化":

    # colored_header(label="数据可视化",description=" ",color_name="violet-90")
    # check NaN
    check_string_NaN(df)
    
    # colored_header(label="数据信息",description=" ",color_name="violet-70")

    # nrow = st.slider("rows", 1, len(df)-1, 5)
    # df_nrow = df.head(nrow)
    # st.write(df_nrow)

    colored_header(label="数据初步统计",description=" ",color_name="violet-30")

    st.write(df.describe())

    # tmp_download_link = download_button(df.describe(), f'数据统计.csv', button_text='download')
    
    # st.markdown(tmp_download_link, unsafe_allow_html=True)

    # colored_header(label="特征变量和目标变量", description=" ",color_name="violet-70")
    
    # target_num = st.number_input('目标变量数量', min_value=1, max_value=10, value=1)
    # col_feature, col_target = st.columns(2)
    target_num = 1
    # features
    features = df.iloc[:,:-target_num]
    # targets
    targets = df.iloc[:,-target_num:]
    # with col_feature:    
    #     st.write(features.head())
    # with col_target:   
    #     st.write(targets.head())

    colored_header(label="特征变量统计分布", description=" ",color_name="violet-30")

    feature_selected_name = st.selectbox('选择特征变量',list(features))

    feature_selected_value = features[feature_selected_name]
    plot = customPlot()
    col1, col2 = st.columns([1,3])
    with col1:  
        with st.expander("绘图参数"):
            options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                        plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('line color',6,5),plot.set_color('bin color',0,6)]
    with col2:
        plot.feature_hist_kde(options_selected,feature_selected_name,feature_selected_value)

    #=========== Targets visulization ==================

    colored_header(label="目标变量统计分布", description=" ",color_name="violet-30")

    target_selected_name = st.selectbox('选择目标变量',list(targets))

    target_selected_value = targets[target_selected_name]
    plot = customPlot()
    col1, col2 = st.columns([1,3])
    with col1:  
        with st.expander("绘图参数"):
            options_selected = [plot.set_title_fontsize(7),plot.set_label_fontsize(8),
                        plot.set_tick_fontsize(9),plot.set_legend_fontsize(10), plot.set_color('line color',6,11), plot.set_color('bin color',0,12)]
    with col2:
        plot.target_hist_kde(options_selected,target_selected_name,target_selected_value)

    #=========== Features analysis ==================

    colored_header(label="特征变量配方（合金成分）", description=" ",color_name="violet-30")

    feature_range_selected_name = st.slider('选择特征变量个数',1,len(features.columns), (1,2))
    min_feature_selected = feature_range_selected_name[0]-1
    max_feature_selected = feature_range_selected_name[1]
    feature_range_selected_value = features.iloc[:,min_feature_selected: max_feature_selected]
    data_by_feature_type = df.groupby(list(feature_range_selected_value))
    feature_type_data = create_data_with_group_and_counts(data_by_feature_type)
    IDs = [str(id_) for id_ in feature_type_data['ID']]
    Counts = feature_type_data['Count']
    col1, col2 = st.columns([1,3])
    with col1:  
        with st.expander("绘图参数"):
            options_selected = [plot.set_title_fontsize(13),plot.set_label_fontsize(14),
                        plot.set_tick_fontsize(15),plot.set_legend_fontsize(16),plot.set_color('bin color',0, 17)]
    with col2:
        plot.featureSets_statistics_hist(options_selected,IDs, Counts)

    # colored_header(label="特征变量在数据集中的分布", description=" ",color_name="violet-30")
    # feature_selected_name = st.selectbox('选择特征变量', list(features),1)
    # feature_selected_value = features[feature_selected_name]
    # col1, col2 = st.columns([1,3])
    # with col1:  
    #     with st.expander("绘图参数"):
    #         options_selected = [plot.set_title_fontsize(18),plot.set_label_fontsize(19),
    #                     plot.set_tick_fontsize(20),plot.set_legend_fontsize(21), plot.set_color('bin color', 0, 22)]
    # with col2:
    #     plot.feature_distribution(options_selected,feature_selected_name,feature_selected_value)

    # colored_header(label="特征变量和目标变量关系", description=" ",color_name="violet-30")
    # col1, col2 = st.columns([1,3])
    # with col1:  
    #     with st.expander("绘图参数"):
    #         options_selected = [plot.set_title_fontsize(23),plot.set_label_fontsize(24),
    #                     plot.set_tick_fontsize(25),plot.set_legend_fontsize(26),plot.set_color('scatter color',0, 27),plot.set_color('line color',6,28)]
    # with col2:
    #     plot.features_and_targets(options_selected,df, list(features), list(targets))
    
    # st.write("### Targets and Targets ")
    # if targets.shape[1] != 1:
    #     colored_header(label="目标变量和目标变量关系", description=" ",color_name="violet-30")
    #     col1, col2 = st.columns([1,3])
    #     with col1:  
    #         with st.expander("绘图参数"):
    #             options_selected = [plot.set_title_fontsize(29),plot.set_label_fontsize(30),
    #                         plot.set_tick_fontsize(31),plot.set_legend_fontsize(32),plot.set_color('scatter color',0, 33),plot.set_color('line color',6,34)]
    #     with col2:
    #         plot.targets_and_targets(options_selected,df, list(targets))
    # st.write('---')


elif select_option == "特征工程":
    with st.sidebar:
        sub_option = option_menu(None, ["特征和目标相关性", "特征重要性"])

    if sub_option == "特征和目标相关性":
        # colored_header(label="特征和目标相关性",description=" ",color_name="violet-90")
        # check_string_NaN(df)
        # colored_header(label="数据信息", description=" ",color_name="violet-70")
        # nrow = st.slider("rows", 1, len(df)-1, 5)
        # df_nrow = df.head(nrow)
        # st.write(df_nrow)

        # colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        # target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)

        target_num = 1
        # col_feature, col_target = st.columns(2)
        
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        # with col_feature:    
        #     st.write(features.head())
        # with col_target:   
        #     st.write(targets.head())
    
        colored_header(label="丢弃与目标的低相关性特征",description=" ",color_name="violet-70")
        fs = FeatureSelector(features, targets)
        plot = customPlot() 
        target_selected_option = st.selectbox('选择特征', list(fs.targets))
        col1, col2 = st.columns([1,3])

        
        with col1:  
            corr_method = st.selectbox("相关性分析方法",["pearson"], key=15)  
            if corr_method != "MIR":
                option_dropped_threshold = 0.1
            if corr_method == 'MIR':
                options_seed = st.checkbox('random state 1024',True)
            with st.expander('绘图参数'):
                options_selected = [plot.set_title_fontsize(11),plot.set_label_fontsize(12),
                    plot.set_tick_fontsize(13),plot.set_legend_fontsize(14),plot.set_color('bin color',19,16)]
            
        with col2:
            target_selected = fs.targets[target_selected_option]
            if corr_method != "MIR":
                corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method).abs()

                fs.judge_drop_f_t([target_selected_option], corr_matrix, option_dropped_threshold)
                
                fs.features_dropped_f_t = fs.features.drop(columns=fs.ops['f_t_low_corr'])
                corr_f_t = pd.concat([fs.features_dropped_f_t, target_selected], axis=1).corr(corr_method)[target_selected_option][:-1]

                plot.corr_feature_target(options_selected, corr_f_t)
                # with st.expander('处理之后的数据'):
                #     data = pd.concat([fs.features_dropped_f_t, targets], axis=1)
                #     st.write(data)
                #     tmp_download_link = download_button(data, f'丢弃与目标的低相关性特征数据.csv', button_text='download')
                #     st.markdown(tmp_download_link, unsafe_allow_html=True)
            else:
                if options_seed:
                    corr_mir  = MIR(fs.features, target_selected, random_state=1024)
                else:
                    corr_mir = MIR(fs.features, target_selected)
                corr_mir = pd.DataFrame(corr_mir).set_index(pd.Index(list(fs.features.columns)))
                corr_mir.rename(columns={0: 'mutual info'}, inplace=True)
                plot.corr_feature_target_mir(options_selected, corr_mir)
        st.write('---')

    elif sub_option == "特征重要性":
        colored_header(label="特征重要性",description=" ",color_name="violet-90")
        file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
        if file is None:
            table = PrettyTable(['上传文件名称', '名称','数据说明'])
            table.add_row(['file_1','dataset','数据集'])
            st.write(table)
        if file is not None:
            df = pd.read_csv(file)
            # 检测缺失值
            check_string_NaN(df)
            colored_header(label="数据信息", description=" ",color_name="violet-70")
            nrow = st.slider("rows", 1, len(df)-1, 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

            target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
            
            col_feature, col_target = st.columns(2)        
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
    
            fs = FeatureSelector(features,targets)

            colored_header(label="选择目标变量", description=" ", color_name="violet-70")

            target_selected_name = st.selectbox('目标变量', list(fs.targets)[::-1])

            fs.targets = targets[target_selected_name]
            
            colored_header(label="Selector", description=" ",color_name="violet-70")

            model_path = './models/feature importance'
            
            template_alg = model_platform(model_path=model_path)

            colored_header(label="Training", description=" ",color_name="violet-70")

            inputs, col2 = template_alg.show()
            # st.write(inputs)

            if inputs['model'] == 'LinearRegressor':
                
                fs.model = LinearR()

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,10,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:
                    fs.LinearRegressor()     
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv ,scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        fig, ax = plt.subplots()
                        ax = plt.plot(cumu_importance, scores,'o-')
                        plt.xlabel("cumulative feature importance")
                        plt.ylabel("r2")
                        st.pyplot(fig)
            elif inputs['model'] == 'LassoRegressor':
                
                fs.model = Lasso(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,10,5)

                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:

                    fs.LassoRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        fig, ax = plt.subplots()
                        ax = plt.plot(cumu_importance, scores,'o-')
                        plt.xlabel("cumulative feature importance")
                        plt.ylabel("r2")
                        st.pyplot(fig)

            elif inputs['model'] == 'RidgeRegressor':

                fs.model = Ridge(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,10,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:
                    fs.RidgeRegressor()     
                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        fig, ax = plt.subplots()
                        ax = plt.plot(cumu_importance, scores,'o-')
                        plt.xlabel("cumulative feature importance")
                        plt.ylabel("r2")
                        st.pyplot(fig)
            elif inputs['model'] == 'LassoRegressor':
                
                fs.model = Lasso(random_state=inputs['random state'])

                with col2:
                    option_cumulative_importance = st.slider('累计重要性阈值',0.0, 1.0, 0.95)
                    Embedded_method = st.checkbox('Embedded method',False)
                    if Embedded_method:
                        cv = st.number_input('cv',1,10,5)
                with st.container():
                    button_train = st.button('train', use_container_width=True)
                if button_train:

                    fs.LassoRegressor()

                    fs.identify_zero_low_importance(option_cumulative_importance)
                    fs.feature_importance_select_show()
                    if Embedded_method:
                        
                        threshold  = fs.cumulative_importance

                        feature_importances = fs.feature_importances.set_index('feature',drop = False)

                        features = []
                        scores = []
                        cumuImportance = []
                        for i in range(1, len(fs.features.columns) + 1):
                            features.append(feature_importances.iloc[:i, 0].values.tolist())
                            X_selected = fs.features[features[-1]]
                            score = CVS(fs.model, X_selected, fs.targets, cv=cv, scoring='r2').mean()

                            cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                            scores.append(score)
                        cumu_importance = np.array(cumuImportance)
                        scores = np.array(scores) 
                        fig, ax = plt.subplots()
                        ax = plt.plot(cumu_importance, scores,'o-')
                        plt.xlabel("cumulative feature importance")
                        plt.ylabel("r2")
                        st.pyplot(fig)

            elif inputs['model'] == 'RandomForestRegressor':
                        
                        fs.model = RFR(criterion = inputs['criterion'], n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                        min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                        n_jobs=inputs['njobs'])
                        with col2:
                            option_cumulative_importance = st.slider('累计重要性阈值',0.5, 1.0, 0.95)
                            Embedded_method = st.checkbox('Embedded method',False)
                            if Embedded_method:
                                cv = st.number_input('cv',1,10,5)
                            
                        with st.container():
                            button_train = st.button('train', use_container_width=True)
                        if button_train:

                            fs.RandomForestRegressor()

                            fs.identify_zero_low_importance(option_cumulative_importance)
                            fs.feature_importance_select_show()

                            if Embedded_method:
                                
                                threshold  = fs.cumulative_importance

                                feature_importances = fs.feature_importances.set_index('feature',drop = False)

                                features = []
                                scores = []
                                cumuImportance = []
                                for i in range(1, len(fs.features.columns) + 1):
                                    features.append(feature_importances.iloc[:i, 0].values.tolist())
                                    X_selected = fs.features[features[-1]]
                                    score = CVS(fs.model, X_selected, fs.targets, cv=cv ,scoring='r2').mean()

                                    cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                                    scores.append(score)
                                cumu_importance = np.array(cumuImportance)
                                scores = np.array(scores) 
                                fig, ax = plt.subplots()
                                ax = plt.plot(cumu_importance, scores,'o-')
                                plt.xlabel("cumulative feature importance")
                                plt.ylabel("r2")
                                st.pyplot(fig)

            st.write('---')



elif select_option == "回归预测":

    # colored_header(label="回归预测",description=" ",color_name="violet-90")
    # file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    # if file is None:
    #     table = PrettyTable(['上传文件名称', '名称','数据说明'])
    #     table.add_row(['file_1','dataset','数据集'])
    #     st.write(table)
    # if file is not None:
        # df = pd.read_csv(file)
        # # 检测缺失值
        # check_string_NaN(df)

    # colored_header(label="数据信息", description=" ",color_name="violet-70")
    # nrow = st.slider("rows", 1, len(df)-1, 5)
    # df_nrow = df.head(nrow)
    # st.write(df_nrow)

    # colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

    # target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
    target_num = 1
    
    # col_feature, col_target = st.columns(2)
    # features
    features = df.iloc[:,:-target_num]
    # targets
    targets = df.iloc[:,-target_num:]
    # with col_feature:    
    #     st.write(features.head())
    # with col_target:   
    #     st.write(targets.head())
# =================== model ====================================
    reg = REGRESSOR(features,targets)

    # colored_header(label="选择目标变量", description=" ", color_name="violet-70")

    target_selected_option = list(reg.targets)[::-1]

    reg.targets = targets[target_selected_option]

    colored_header(label="Regressor", description=" ",color_name="violet-30")

    model_path = './models/regressors'

    template_alg = model_platform(model_path)

    inputs, col2 = template_alg.show()


    if inputs['model'] == 'RandomForestRegressor':
        with col2:
            with st.expander('Operator'):
                # operator = st.selectbox('data operator', ('train test split','cross val score','leave one out','oob score'))
                operator = st.selectbox('data operator', (['leave one out']))
                # if operator == 'train test split':
                #     inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                #     reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                # elif operator == 'cross val score':
                #     cv = st.number_input('cv',1,10,5)

                if operator == 'leave one out':
                    loo = LeaveOneOut()

                # elif operator == 'oob score':
                #     inputs['oob score']  = st.selectbox('oob score',[True], disabled=True)
                #     inputs['warm start'] = True

        colored_header(label="Training", description=" ",color_name="violet-30")
        with st.container():
            button_train = st.button('Train', use_container_width=True)
        
        if button_train:
            # if operator == 'train test split':

            #     reg.model = RFR( n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
            #                                     min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
            #                                     n_jobs=inputs['njobs'])
                
            #     reg.RandomForestRegressor()

            #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
            #     result_data.columns = ['actual','prediction']
            #     plot_and_export_results(reg, "RFR")


            # elif operator == 'cross val score':

            #     reg.model = RFR(n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
            #                                 min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
            #                                 n_jobs=inputs['njobs'])

            #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
    
            #     plot_cross_val_results(cvs, "RFR_cv") 

            # elif operator == 'oob score':

            #     reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
            #                                 min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
            #                                 n_jobs=inputs['njobs'])
            
            #     reg_res  = reg.model.fit(reg.features, reg.targets)
            #     oob_score = reg_res.oob_score_
            #     st.write(f'oob score : {oob_score}')

            # if operator == 'leave one out':

            reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'])
            
            export_loo_results(reg, loo, "RFR_loo")

    if inputs['model'] == 'SupportVector':

        with col2:
            with st.expander('Operator'):

                # preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                preprocess = st.selectbox('data preprocess',['StandardScaler'])

                # operator = st.selectbox('operator', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
                operator = st.selectbox('operator', ['leave one out'], label_visibility='collapsed')
                # if operator == 'train test split':
                #     inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                #     if preprocess == 'StandardScaler':
                #         reg.features = StandardScaler().fit_transform(reg.features)
                #     if preprocess == 'MinMaxScaler':
                #         reg.features = MinMaxScaler().fit_transform(reg.features)
                    
                #     reg.features = pd.DataFrame(reg.features)    
                    
                #     reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                # elif operator == 'cross val score':
                #     if preprocess == 'StandardScaler':
                #         reg.features = StandardScaler().fit_transform(reg.features)
                #     if preprocess == 'MinMaxScaler':
                #         reg.features = MinMaxScaler().fit_transform(reg.features)
                #     cv = st.number_input('cv',1,10,5)

                if operator == 'leave one out':
                    if preprocess == 'StandardScaler':
                        reg.features = StandardScaler().fit_transform(reg.features)
                    if preprocess == 'MinMaxScaler':
                        reg.features = MinMaxScaler().fit_transform(reg.features)
                    loo = LeaveOneOut()
        colored_header(label="Training", description=" ",color_name="violet-30")
        with st.container():
            button_train = st.button('Train', use_container_width=True)
        if button_train:
            # if operator == 'train test split':

            #     reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                
            #     reg.SupportVector()

            #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
            #     result_data.columns = ['actual','prediction']
                
            #     plot_and_export_results(reg, "SVR")

            # elif operator == 'cross val score':

            #     reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

            #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            #     plot_cross_val_results(cvs, "SVR_cv")  


            if operator == 'leave one out':

                reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
            
                export_loo_results(reg, loo, "SVR_loo")           
                          
    # if inputs['model'] == 'LassoRegressor':

    #     with col2:
    #         with st.expander('Operator'):

    #             preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

    #             operator = st.selectbox('', ('train test split','cross val score', 'leave one out'), label_visibility='collapsed')
    #             if operator == 'train test split':
    #                 inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
    #                 if preprocess == 'StandardScaler':
    #                     reg.features = StandardScaler().fit_transform(reg.features)
    #                 if preprocess == 'MinMaxScaler':
    #                     reg.features = MinMaxScaler().fit_transform(reg.features)
                    
    #                 reg.features = pd.DataFrame(reg.features)    
                    
    #                 reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    
    #             elif operator == 'cross val score':
    #                 if preprocess == 'StandardScaler':
    #                     reg.features = StandardScaler().fit_transform(reg.features)
    #                 if preprocess == 'MinMaxScaler':
    #                     reg.features = MinMaxScaler().fit_transform(reg.features)
    #                 cv = st.number_input('cv',1,10,5)

    #             elif operator == 'leave one out':
    #                 if preprocess == 'StandardScaler':
    #                     reg.features = StandardScaler().fit_transform(reg.features)
    #                 if preprocess == 'MinMaxScaler':
    #                     reg.features = MinMaxScaler().fit_transform(reg.features)
    #                 loo = LeaveOneOut()
        
    #     colored_header(label="Training", description=" ",color_name="violet-30")
    #     with st.container():
    #         button_train = st.button('Train', use_container_width=True)
    #     if button_train:
    #         if operator == 'train test split':

    #             reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                
    #             reg.LassoRegressor()

    #             result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
    #             result_data.columns = ['actual','prediction']
                
    #             plot_and_export_results(reg, "LassoR")

    #         elif operator == 'cross val score':

    #             reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])

    #             cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)

    #             plot_cross_val_results(cvs, "LassoR_cv")   

    #         elif operator == 'leave one out':

    #             reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
            
    #             export_loo_results(reg, loo, "LassoR_loo")

    if inputs['model'] == 'MLPRegressor':

        with col2:
            with st.expander('Operator'):

                preprocess = st.selectbox('data preprocess',['StandardScaler'])

                operator = st.selectbox('operator', (['leave one out']), label_visibility='collapsed')
                # if operator == 'train test split':
                #     inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                #     if preprocess == 'StandardScaler':
                #         reg.features = StandardScaler().fit_transform(reg.features)
                #     if preprocess == 'MinMaxScaler':
                #         reg.features = MinMaxScaler().fit_transform(reg.features)
                    
                #     reg.features = pd.DataFrame(reg.features)    
                    
                #     reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    
                # elif operator == 'cross val score':
                #     if preprocess == 'StandardScaler':
                #         reg.features = StandardScaler().fit_transform(reg.features)
                #     if preprocess == 'MinMaxScaler':
                #         reg.features = MinMaxScaler().fit_transform(reg.features)
                #     cv = st.number_input('cv',1,10,5)
                
                if operator == 'leave one out':
                    if preprocess == 'StandardScaler':
                        reg.features = StandardScaler().fit_transform(reg.features)
                    if preprocess == 'MinMaxScaler':
                        reg.features = MinMaxScaler().fit_transform(reg.features)
                    loo = LeaveOneOut()              
        colored_header(label="Training", description=" ",color_name="violet-30")
        with st.container():
            button_train = st.button('Train', use_container_width=True)
        if button_train:
            # if operator == 'train test split':

            #     reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'] 
            #                         ,  max_iter=inputs['max iter'], random_state=inputs['random state'])
            #     reg.MLPRegressor()

            #     result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
            #     result_data.columns = ['actual','prediction']
            #     plot_and_export_results(reg, "MLP")

            # elif operator == 'cross val score':

            #     reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
            #                             batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
            #                             random_state=inputs['random state'])
                
            #     cvs = CV(reg.model, reg.features, reg.targets, cv = cv, scoring=make_scorer(r2_score), return_train_score=False, return_estimator=True)
            #     plot_cross_val_results(cvs, "MLP_cv") 

            if operator == 'leave one out':

                reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                         max_iter=inputs['max iter'],random_state=inputs['random state'])
                                        
            
                export_loo_results(reg, loo, "MLP_loo")

        st.write('---')

elif select_option == "可解释性机器学习":

    # colored_header(label="可解释性机器学习",description=" ",color_name="violet-90")

    # file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    # if file is None:
    #     table = PrettyTable(['上传文件名称', '名称','数据说明'])
    #     table.add_row(['file_1','dataset','数据集'])
    #     st.write(table)        
    # if file is not None:
    #     df = pd.read_csv(file)
        # 检测缺失值
        # check_string_NaN(df)
        # colored_header(label="数据信息", description=" ",color_name="violet-70")
        # nrow = st.slider("rows", 1, len(df)-1, 5)
        # df_nrow = df.head(nrow)
        # st.write(df_nrow)

        # colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        # target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)
        target_num = 1
        # col_feature, col_target = st.columns(2)
            
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        # with col_feature:    
        #     st.write(features.head())
        # with col_target:   
        #     st.write(targets.head())


        colored_header(label="Shapley value",description=" ",color_name="violet-70")

        fs = FeatureSelector(features, targets)

        target_selected_option = st.selectbox('choose target', list(fs.targets))
        fs.targets = fs.targets[target_selected_option]
        # regressor = st.selectbox('tree',['linear','kernel','sampling'])
        reg = RFR()
        X_train, X_test, y_train, y_test = TTS(fs.features, fs.targets, random_state=0) 
        
        test_size = 0.01 
        random_state = st.checkbox('random state 42',True)
        if random_state:
            random_state = 42
        else:
            random_state = None
            
        fs.Xtrain,fs.Xtest, fs.Ytrain, fs.Ytest = TTS(fs.features,fs.targets,test_size=test_size,random_state=random_state)
        reg.fit(fs.Xtrain, fs.Ytrain)

        explainer = shap.TreeExplainer(reg)
        
        shap_values = explainer(fs.features)

        # colored_header(label="SHAP Feature Importance", description=" ",color_name="violet-30")
        # nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1])
        # st_shap(shap.plots.bar(shap_values, max_display=nfeatures))
        # st.write(shap_values)
        # colored_header(label="SHAP Feature Cluster", description=" ",color_name="violet-30")
        # clustering = shap.utils.hclust(fs.features, fs.targets)
        # clustering_cutoff = st.slider('clustering cutoff', 0.0,1.0,0.5)
        # nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1], key=2)
        # st_shap(shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=clustering_cutoff, max_display=nfeatures))

        # colored_header(label="SHAP Beeswarm", description=" ",color_name="violet-30")
        # rank_option = st.selectbox('rank option',['max','mean'])
        # max_dispaly = st.slider('max display',2, fs.features.shape[1],fs.features.shape[1])
        # if rank_option == 'max':
        #     st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.max(0), max_display =max_dispaly))
        # else:
        #     st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.mean(0), max_display =max_dispaly))

        colored_header(label="SHAP Dependence", description=" ",color_name="violet-30")
        
        shap_values = explainer.shap_values(fs.features) 
        list_features = fs.features.columns.tolist()
        feature = st.selectbox('feature',list_features)
        interact_feature = st.selectbox('interact feature', list_features)
        st_shap(shap.dependence_plot(feature, shap_values, fs.features, display_features=fs.features,interaction_index=interact_feature))
               

elif select_option == "模型推理":
    
    # colored_header(label="模型推理",description=" ",color_name="violet-90")
    file = st.file_uploader("Upload `.csv`file ",  label_visibility="collapsed")
    # if len(file) < 2:
    #     table = PrettyTable(['上传文件名称', '名称','数据说明'])
    #     table.add_row(['file_1','data set','数据集'])
    #     table.add_row(['file_2','model','模型'])
    #     st.write(table)
    # elif len(file) == 2:
    # df = pd.read_csv(file[0])
    # model_file = file[1]
    if file:
        df =  pd.read_csv(file)
        # check_string_NaN(df)

        # colored_header(label="数据信息", description=" ",color_name="violet-70")
        # nrow = st.slider("rows", 1, len(df)-1, 5)
        # df_nrow = df.head(nrow)
        # st.write(df_nrow)

        # colored_header(label="特征变量和目标变量",description=" ",color_name="violet-70")

        # target_num = st.number_input('目标变量数量',  min_value=1, max_value=10, value=1)

        target_num = 1
        col_feature, col_target = st.columns(2)
        # features
        # features = df.iloc[:,:-target_num]
        # targets
        # targets = df.iloc[:,-target_num:]
        # with col_feature:    
        #     st.write(features.head())
        # with col_target:   
        #     st.write(targets.head())    
        # colored_header(label="选择目标变量", description=" ", color_name="violet-70")

        # target_selected_option = st.selectbox('target', list(targets)[::-1])
        # targets = targets[target_selected_option]
        preprocess = st.selectbox('data preprocess',['StandardScaler'])
        # if preprocess == 'StandardScaler':
        features = StandardScaler().fit_transform(df)
        # elif preprocess == 'MinMaxScaler':
        #     features = MinMaxScaler().fit_transform(features)
        model_path = './data/svm.pickle'
        # model_file = pickle.dumps(model_path)
        # model = pickle.loads(model_path)
        with open(model_path,'rb') as f:  
            model = pickle.load(f)  #将模型存储在变量clf_load中  
            # print(clf_load.predict(X[0:1000])) #调用模型并预测结果
            prediction = model.predict(features)
        st.write("预测结果")
        result_data = pd.DataFrame(prediction)
        # st.write(result_data)
        result_data.columns = ['TOL预测值']
   
        # plot = customPlot()
        # plot.pred_vs_actual(df, prediction)
        # r2 = r2_score(df, prediction)
        # st.write('R2: {}'.format(r2))
        # result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
        # result_data.columns = ['actual','prediction']
        # with st.expander('预测结果'):
        st.write(result_data)
        tmp_download_link = download_button(result_data, f'预测结果.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        st.write('---')
