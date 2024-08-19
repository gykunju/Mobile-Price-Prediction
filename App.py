import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from Model import stack_model

st.write('''
# Mobile Price Prediction   
''')


st.write('The Mobile Price range data is from kaggle...https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv')
st.write('The data gives a price range description for a mobile device based on certain mobile characteristics given')
st.write('The task was to create a price range prediction model to inform individuals on the price range of their mobile device based on their device hardware and software characteristics')

train_data = pd.read_csv('./train.csv')

st.write('''
## Data Set:
         ''')

st.expander('Data').dataframe(train_data)

col1, col2, col3 = st.columns(3)
col1.metric('Entries', train_data.count()[0])
col2.metric('Target Classes', train_data['price_range'].nunique())
col3.metric('Features', len(train_data.drop(columns=['price_range']).columns.to_list()))

st.write('''
### Description of features:

 1- battery_power: Total energy a battery can store in one time measured in (mAh)

 2- blue: Has bluetooth or not

 3- clock_speed: Speed at which microprocessor executes instructions

 4- dual_sim: Has dual sim support or not

 5- fc: Front camera (Megapixels)

 6- four_g: Has 4G or not

 7- int_memory: Internal memory in (Gigabytes)

 8- m_dep: Mobile depth in (Cm)

 9- mobile_wt: Weight of mobile phone

 10- pc: Primary camera (Megapixels)

 11- px_height: Pixel resolution height

 12- px_width: Pixel resolution width

 13- ram: Random access memory in (Megabytes)

 14- sc_h: Screen height of mobile in (Cm)

 15- sc_w: Screen width of mobile in (Cm)

 16- talk_time: Longest time that a single battery charge will last when you are constantly talking on the phone

 17- three_g: Has 3G or not

 18- touch_screen: Has touch screen or not

 19- wifi: Has wifi or not

 20- n_cores: Number of cores of processor

 21- price_range: This is the Target variable with value of 0: (Low Cost), 1: (Medium Cost), 2: (High Cost), and 3: (Very High Cost)
         ''')


st.write('''

## EDA
         ''')



### PIE CHART - target distribution
st.subheader('Target Distribution')

col1, col2 = st.columns([0.7,0.3])
fig, ax = plt.subplots(figsize=(10,10))
grouped = train_data.groupby('price_range').count()['blue']
plt.pie(grouped.values, labels=['Low Cost', 'Meduim Cost', 'High Cost', 'Very High Cost'], autopct='%1.1f%%', wedgeprops = { 'linewidth' : 4, 'edgecolor' : 'white' })
col1.pyplot(fig)
st.write('- Data is evenly distributed on the target')


### Histplots


# plt.suptitle('Histograms of Features')
# st.pyplot(train_data.hist(bins=30, figsize=(20, 20)))

### CORRELATION HEATMAP - feature and target relations
st.subheader('Data Relations')
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(train_data.corr(), annot=True)
plt.title('Correlation Heatmap of the data')
st.pyplot(fig)
st.write('- For the target (Price range), The ram feature seems to have a strong relationship with it. Additionally, there are some recognisable relation with the px_width, px_height and battery power')
st.write('- Furthemore there seems to be some significant correlation between the features pc and fc, px_width and px_height and three_g and four_g')


### BOXPLOTS - distribution of data
st.subheader('Distribution of features')
fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data=train_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Boxplots of Features')
st.pyplot(fig)

st.write('- We get a clear view of the discrete and continuous features and their distribution in the data')
st.write('- ram has the widest range of values for the continuous features')


### RAM and Price Range


st.subheader('Ram and Price range Analysis')
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='price_range', y='ram', data=train_data, palette='coolwarm')
plt.title('RAM vs. Price Range')
st.pyplot(fig)

st.write('- The ram is seen to increase with the price range insinuating that they are strongly related ')



### Pixel Height and Pixel Width
st.subheader('Pixel Height and Pixel Width')
fig = plt.figure(figsize=(10,10))
scatter = plt.scatter(data=train_data, x='px_height', y='px_width', c='price_range', cmap='coolwarm')
legend1 = plt.legend(*scatter.legend_elements(), title="Price Range")
plt.gca().add_artist(legend1)
plt.title('Relation between px_width and px_height')
st.pyplot(fig)

st.write('- There is a positive correlation between px_height and px_width since the px_width increases with pixel_height')
st.write('- Insinuates that higher px_height tend to have higher px-width')



### Battery and Talk time
st.subheader('Battery Power and Talk time Analysis')
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='battery_power', y='talk_time', hue='price_range', data=train_data, palette='coolwarm')
plt.title('Battery Power vs. Talk Time')
st.pyplot(fig)

st.write('- Battery power and talk time have no relation to each other')
st.write('- There are instances where a very highly priced phone has poor battery power and poor talk time')



### Cores and Clock Speed
st.subheader('Number of Cores and Clock speed Analysis')
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='n_cores', y='clock_speed', hue='price_range', data=train_data, palette='coolwarm')
plt.title('Number of cores vs. Clock Speed')
st.pyplot(fig)

st.write('- Number of Cores and clock speed have no relation to each other')


### three_g and four_g
st.subheader('3G and 4G Support')
contingency_table = pd.crosstab(train_data['four_g'], train_data['three_g'])
fig = plt.figure(figsize=(4,4))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('3G')
plt.ylabel('4G')
plt.title('Relationship between 4G and 3G Support')
st.pyplot(fig)

st.write('- Insinuates that a device cannot lack 3G and 4G connectivity')

### 
st.subheader('RAM, Battery Power and Price Range')
fig = plt.figure(figsize=(10, 8))
scatter = plt.scatter(train_data['ram'], train_data['battery_power'], c=train_data['price_range'], cmap='coolwarm')
plt.xlabel('RAM')
plt.ylabel('Battery Power')
plt.title('RAM vs. Battery Power vs. Price Range')
plt.legend()

legend1 = plt.legend(*scatter.legend_elements(), title="Price Range")
plt.gca().add_artist(legend1)

st.pyplot(fig)

st.write('- RAM still clearly shows up as the major contributor in determining the price range')
st.write('- The price range seems to increase with increase in RAM ')




### Area Calculation

# data = train_data.copy()
# data['Area'] = data['sc_h'] * data['sc_w']


# ### Area to weight ratio
# fig = plt.figure(figsize=(10,10))

# plt.scatter(data=data, x='Area', y='mobile_wt', c='price_range', cmap='coolwarm')
# plt.title('Screen Area to Weight Ratio')

# st.pyplot(fig)

### PREDICTION MODEL
st.subheader('Prediction Model')
st.write('The Data was scaled using a standard scaler and split into 80% for the training and 20% for the testing')
st.write(''' 
         ### Models Used were: 
         - Logistic Regression
         - Support Vector Classifier
         ''')

st.write('The logistic Regressor and SVC proved to be the best performing prediction models')
st.write('##### - Logistic Regressor (penalty=None, max_iter=1000)')

col1, col2 = st.columns(2)
col1.metric('Accuracy', 1.00, 'Train')
col2.metric('Accuracy', 0.97,'Test')



st.write('##### - SVC (degree=1, kernel=linear, C=0.7)')
col1, col2 = st.columns(2)
col1.metric('Accuracy', 0.97, 'Train')
col2.metric('Accuracy', 0.96, 'Test')


st.write('A stacking ensemble was used with the estimators as The logistic regressor and the SVC and the final estimator being the logistic regressor')
st.write('The decision was based off the fact that the stack provided less misclassification compared to the other models')
st.write('##### - Stack')


st.subheader('Predict')
col1, col2 = st.columns(2)
col1.metric('Accuracy', 0.98, 'Train')
col2.metric('Accuracy', 0.97, 'Test')

col1, col2, col3 = st.columns(3)
battery = col1.number_input('Battery Power (mAh)', value=500)
ram = col2.number_input('RAM (Mbs)', value=2000)
weight = col3.number_input('Weight (grams)', value=100)


m_dep = st.select_slider('Mobile Depth', np.arange(0.1,1.1,0.1), value=0.1)


col1, col2, col3 = st.columns(3)
talk_time = col1.number_input('Talk Time')
fc = col2.number_input('Front Camera Pixels (MPs)', value=2)
pc = col3.number_input('Primary Camera Pixels (MPs)', value=2)


n_cores = st.select_slider('Number of Cores', [1,2,3,4,5,6,7,8])


col1,col2,col3 = st.columns(3)
int_memory = col1.number_input('internal Memory (GB)')
px_width = col2.number_input('Pixel Width')
px_height = col3.number_input('Pixel Height')


clock_speed = st.number_input('Clock Speed')


col1, col2, col3, col4 = st.columns(4)
three_g = col1.selectbox('Has 3G', ['Yes', 'No'])
four_g = col2.selectbox('Has 4G', ['Yes', 'No']) if three_g == 'Yes' else 'No'
sc_h = col3.number_input('Screen Height (cm)')
sc_w = col4.number_input('Screen Width (cm)')


col1, col2, col3, col4 = st.columns(4)
bluetooth = col1.selectbox('Has Bluetooth', ['Yes', 'No'])
dual_sim = col2.selectbox('Has Dual Sim', ['Yes', 'No'])
touch_screen = col3.selectbox('Has Touch screen', ['Yes', 'No'])
wifi = col4.selectbox('Has wifi', ['Yes', 'No'])


def categorical(val):
    return 1 if val == 'Yes' else 0


### battery, categorical(bluetooth), clock_speed, categorical(dual_sim), fc, categorical(four_g), int_memory, m_dep, weight, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, categorical(three_g), categorical(touch_screen), categorical(wifi)

def Predict():
    row = np.array([battery, categorical(bluetooth), clock_speed, categorical(dual_sim), fc, categorical(four_g), int_memory, m_dep, weight, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, categorical(three_g), categorical(touch_screen), categorical(wifi)])
    X = pd.DataFrame([row], columns=train_data.drop(columns=['price_range']).columns.to_list())
    prediction = stack_model.predict(X)
    if prediction == 0:
        st.session_state['answer'] = 'Low Cost'
        st.session_state['range'] = '0 - 10,000'
    elif prediction == 1:
        st.session_state['answer'] = 'Medium Cost'
        st.session_state['range'] = '10,001 - 40,000'
    elif prediction == 2:
        st.session_state['answer'] = 'High Cost'
        st.session_state['range'] = '40,001 - 70,000'
    else:
        st.session_state['answer'] = 'Very High Cost'
        st.session_state['range'] = 'upwards of 70,000'

if 'answer' not in st.session_state:
    st.session_state['answer'] = 'N/A'



st.button('Predict Range', on_click=Predict)

st.success(f'Price Range: {st.session_state["answer"]} ({st.session_state["range"] if "range" in st.session_state else None})')


