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