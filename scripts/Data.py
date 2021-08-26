class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def remove_outlier(self, df):
        low = .05
        high = .95
        quant_df = df.quantile([low, high])
        for name in list(df.columns):
            if is_numeric_dtype(df[name]):
                df = df[(df[name] >= quant_df.loc[low, name]) 
                    & (df[name] <= quant_df.loc[high, name])]
        return df

    def transform(self,data):
        scaler=MinMaxScaler()
        enc = OneHotEncoder(handle_unknown='ignore')
        cat=[]
        non_cat=[]
        
        if 'y' in data.columns:
            data.y=data.y.map({'no':0,'yes':1})
            for i in data:
                if data[i].dtype==object:
                    cat.append(i)
                else:
                    non_cat.append(i)
            enc_df = pd.DataFrame(enc.fit_transform(data[cat]).toarray())
            # merge with main data on key values
            newdata=data[non_cat]
            newdata= newdata.join(enc_df)
            newdata=self.remove_outlier(newdata)
            new_data_scaled=scaler.fit_transform(newdata.drop(['y'],axis=1))
            new_data_scaled=pd.DataFrame(new_data_scaled,columns=newdata.drop(['y'],axis=1).columns)
        else:
            for i in data:
                if data[i].dtype==object:
                    cat.append(i)
                else:
                    non_cat.append(i)
            enc_df = pd.DataFrame(enc.fit_transform(data[cat]).toarray())
            # merge with main data on key values
            newdata=data[non_cat]
            newdata= newdata.join(enc_df)
            newdata=self.remove_outlier(newdata)
            new_data_scaled=scaler.fit_transform(newdata)
            new_data_scaled=pd.DataFrame(new_data_scaled,columns=newdata.columns)
        pca = PCA(n_components = 23)
        pca.fit(new_data_scaled)
        reduced = pca.transform(new_data_scaled)
        reduced=pd.DataFrame(reduced,columns=range(reduced.shape[1]))
        reduced['y']=newdata['y'].values
        return reduced
        
        
        
        

