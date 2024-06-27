# 은행 채무 불이행자 예측 프로젝트

Machine Learning 모델 및 XAI

## 프로젝트 구조

---

### **1. 프로젝트 개요**

은행 대출 디폴트(채무불이행) 여부를 예측하기 위한 머신러닝 모델 구축 프로젝트

- **사용 데이터**
    - Kaggle 제공 데이터
    - 고객의 대출 정보, 연령, 지역 등을 포함한 데이터셋
    - 148670 rows × 34 columns
    - 종속 변수 (0 : 채무 이행자, 1 : 채무 불이행자)
- **사용 모델**
    1. KNN Classifier
    2. Ensemble Model
        - Random Forest
        - XGBoost
        - Light GBM
        - CatBoost
    3. Deep Neural Network (DNN)
- **XAI:** lime, shap
- **환경:** Jupyter Notebook(Local)

### **2. 데이터 전처리**

불필요한 변수 제거, 형변환, 결측치 처리, 변수 형태에 따른 Encoding 적용, Correlation Analysis 등

```python
data.drop(['ID','year'],axis=1,inplace=True)
target=data['Status']
data.drop(columns=['Status'],axis=1,inplace=True)

...

for col in category_data.columns:
  if(col=='age'):
    continue
  category_data = pd.get_dummies(data=category_data,columns=[col])
  
...

ir_reg = LinearRegression()
imputer = IterativeImputer(estimator=ir_reg,verbose=2,max_iter=20,tol=0.001,imputation_order='roman')
data = pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
```

### **3. 모델 구성 및 학습**

Grid Search 함수 구성

- KNN Classifier
    
    ```python
    def knn_model_tuned(X_train,y_train):
    	k_range = range(1,11)
    	param_grid = dict(n_neighbors=k_range)
    	clf = KNeighborsClassifier()
    	grid = GridSearchCV(clf, param_grid, cv=4, scoring = 'accuracy')   
    	grid.fit(X_train,y_train)
    	knn_model = grid.best_estimator_
    	return knn_model
    ```
    
- Random Forest
    
    ```python
    def rf_model_tuned(X_train,y_train):
    	grid_search = {'n_estimators':[90,100,110],
    		              'max_depth' :[40,41,42],
    		              'min_samples_split' :[2,3,4]}
    	clf=RandomForestClassifier(random_state=2022)
    	grid=GridSearchCV(estimator=clf,param_grid=grid_search,cv=4,n_jobs=-1,verbose=5)
    	grid.fit(X_train,y_train)
    	rf_model=grid.best_estimator_
    	return rf_model
    ```
    
- XGBoost
    
    ```python
    def xgb_model_tuned(X_train,y_train):
      grid_search = {'max_depth': [5,6,7],
    	               'min_child_weight': [1,2],
    	               'learning_rate': [0.1,0.2,0.3],
    	               'n_estimators': [50,100]}
      clf=XGBClassifier(objective='binary:logistic',eval_metric='auc',random_state=2022)
      grid=GridSearchCV(estimator=clf,param_grid=grid_search,cv=5,verbose=5,n_jobs=-1)
      grid.fit(X_train,y_train)
      xgb_model=grid.best_estimator_
      return xgb_model
    ```
    
- LightGBM
    
    ```python
    def lgb_model_tuned(X_train,y_train):
      grid_search = {'num_leaves': [11,21,31],
    	               'min_child_samples': [15, 20, 25],
    	               'learning_rate': [0.1,0.07,0.05],
    	               'n_estimators': [50,100]}
      clf=LGBMClassifier()
      grid=GridSearchCV(estimator=clf,param_grid=grid_search,cv=5,verbose=5,n_jobs=-1)
      grid.fit(X_train,y_train)
      lgb_model=grid.best_estimator_
      return lgb_model
    ```
    
- CatBoost
    
    ```python
    def cat_model_tuned(X_train,y_train):
      grid_search = {'max_depth': [5,6,7],
    		             'learning_rate': [0.1,0.07,0.05],
    		             'iterations': [100,500,1000]}
      clf=CatBoostClassifier(silent=True, random_seed=2022)
      grid=GridSearchCV(estimator=clf,param_grid=grid_search,cv=5,verbose=5,n_jobs=-1)
      grid.fit(X_train,y_train)
      cat_model=grid.best_estimator_
      return cat_model
    ```
    
- DNN
    
    ```python
    def dnn_epoch_100(X_train,y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2022)
        tf.random.set_seed(2022)
        initializer = tf.keras.initializers.GlorotUniform(seed=42)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, baseline=0.9)
        check_point = ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min',save_best_only=True)
        
        model = Sequential(name='DNN')
        model.add(Dense(300, input_shape = (X_train.shape[1],), activation='relu', kernel_initializer=initializer, name = 'Input_layer'))
        model.add(Dense(300, activation='relu', kernel_initializer=initializer, name = 'Hidden_layer_1'))
        model.add(Dense(300, activation='relu', kernel_initializer=initializer, name = 'Hidden_layer_2'))
        model.add(Dense(300, activation='relu', kernel_initializer=initializer, name = 'Hidden_layer_3'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer, name='Output_layer'))
    
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics='accuracy')
        
        model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val),callbacks=[early_stopping,check_point])
        
        best_model = load_model('best_model.h5')
        
        return best_model
    ```
    

### **4. 최적 모델 선정 및 XAI**

- HyperParameter Tuning 이후 F1 Score 비교
    - KNN Classifier - 0.5283
    - Random Forest - 0.9220
    - XGBoost - 0.9983
    - LightGBM - 0.9932
    - CatBoost - 0.9977
    - DNN - 0.9680
    
    → 정확도, 학습속도, 과적합 위험성을 고려해 XGBoost 모델 선정
    
- XAI
    - Lime
      
      ![Untitled](https://github.com/odsyjr2/Bank_Loan_Default/assets/44573776/f57714f5-c647-4717-a876-267d305a28c7)

      ![Untitled 1](https://github.com/odsyjr2/Bank_Loan_Default/assets/44573776/6254a5f2-7702-40a2-9578-fff946665a35)

    
    - Shap
      
      ![Untitled 2](https://github.com/odsyjr2/Bank_Loan_Default/assets/44573776/c878fba1-f017-45d5-b615-8d24f106a9c2)
    
    
    → 영향력 있는 변수 : **Income, Credit_type_EQUI, Credit_Worthiness**   
    
    - 수익이 적을수록 채무 불이행 가능성 증가
    - Credit Score 산정기관이 “EQUI”인 경우 채무 불이행 가능성 증가
    - 대출 기관이 “I2”인 경우 채무 불이행 가능성 증가
    

### **5. 결론**

본 프로젝트는 머신러닝 모델을 활용하여 은행 대출 디폴트 여부를 예측하는 것을 목표로 하였습니다. 다양한 모델을 비교한 결과, XGBoost 모델이 가장 우수한 성능을 보여 최종 모델로 선정되었습니다.

XGBoost 모델은 고객의 수익, 크레딧 점수 산정 기관, 대출 기관 등 여러 변수를 중요하게 고려하여 채무 불이행 가능성을 정확하게 예측할 수 있음을 확인하였습니다. 특히, 수익이 적을수록, 크레딧 점수 산정 기관이 “EQUI”인 경우, 대출 기관이 “I2”인 경우 채무 불이행 가능성이 증가하는 경향을 보였습니다.

이러한 예측 결과는 은행 및 금융 기관들이 대출 신청자의 신용 위험을 효과적으로 평가하고 관리하는 데 큰 도움이 될 것입니다. 나아가, 이러한 예측 모델을 통해 금융 기관은 채무 불이행 위험을 사전에 인지하고 대응함으로써 재정적 안정성을 높일 수 있을 것입니다. 
