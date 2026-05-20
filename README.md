### Our goal 
Our goal is to give predictions for the energy consumption for tommorow.
Given past energy consumption data and forecast data (such as time, weather, hollidays, events...) we will give an energy consumption estimate for each time slot for the next day.

Our forecasts will be updated and be more accurate as we receive new and more precise forecast data.


### Overview  

- We collect historical data about energy consumption with other features that we'll able to obtain before the predictions of our model
- We merge all these data in one single dataset + cleaning this dataset. Then we train the model (we'll put the collecting, cleaning and training processes in different python script in order to create a pipeline that will automate the project)
- Once we have finished to train the model, we'll collect the data and the forecast data (weather, events...) that will be given to the model to give predictions for every time slot for the next day.

### Source of the data for predictions  

From : https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/telecharger-indicateurs

We download the dataset called " **En-cours mensuel temps réel** ".  
This dataset provides us recent data covering the period from the beggining of the current month to the current date.



### Dataset that will be used for the training
- Historical data of observed electricity consumption
- Historical data of observed metrics about the weather of France's biggest cities
- Historical calender of school holidays (depending on the zone A, B, C), national and public holidays ('jours fériés', strikes...)

### Dataset that will be given to the model after the training
- The time slots of the next day (we'll observe if the next day is special (any holiday))
- The weather metrics of the next day for each time slots of the major cities in France

| City	| Why include it |
| ------|----------------|
|Paris	| biggest demand center|
|Lille	| northern colder zone|
|Marseille | Mediterranean south|
|Lyon | major inland urban hub|
|Toulouse | southwest climate|
|Bordeaux | Atlantic southwest|
|Nantes	| western oceanic climate|
|Strasbourg	| continental east|
|Montpellier | Mediterranean variation|
|Nice | Riviera coastal climate|
|Grenoble | alpine / mountain influence|
|Rennes	| northwest oceanic regime|


We'll use the weather metrics of these main cities instead of their respective departments because it'll be easier for us to collect the forecast weather metrics for the predictions. There are a ton of APIs that provide those services.


### Dataset features  

For the dataset `conso `, we'll keep few columns and also add multiple ones.  
Default columns : 
- Consommation

  
New columns : 
- Lagged columns : new features giving informations about the consommation during the previous time units (t-15min, t-1, t-4, t-24, t-96)
- day of the week
- month
- season
- weather data
- calendar (school and public holidays)