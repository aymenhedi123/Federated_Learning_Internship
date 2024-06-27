from sklearn.preprocessing import LabelEncoder
def drop_null(df):
    return df.dropna()


def drop_duplicated(df):
    return df.drop_duplicates()

def drop_useless_f(df):
    label_encoder = LabelEncoder()
    df.drop(columns=['ID','oral'],inplace=True)
    df['gender']=label_encoder.fit_transform(df['gender'])
    df['tartar']=label_encoder.fit_transform(df['tartar'])
    return df

def x_y_split(df):
    # Our target is called 'y'
    X = df.drop('y',axis=1).values
    y = df['y'].values
    return X,y
