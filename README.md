# used_cars
# Analysis of German Used Car Sales from eBay Classifieds

The aim of this project is to clean the data and analyze the included used car listings. The listings are from Germany and some relevant information / values are in German. Our data cleaning process included making conversions from German to English. 

The data dictionary provided with data is as follows:

1. dateCrawled - When this ad was first crawled. All field-values are taken from this date.
2. name - Name of the car.
3. seller - Whether the seller is private or a dealer.
4. offerType - The type of listing
5. price - The price on the ad to sell the car.
6. abtest - Whether the listing is included in an A/B test.
7. vehicleType - The vehicle Type.
8. yearOfRegistration - The year in which the car was first registered.
9. gearbox - The transmission type.
10. powerPS - The power of the car in PS (horse power).
11. model - The car model name.
12. kilometer - How many kilometers the car has driven.
13. monthOfRegistration - The month in which the car was first registered.
14. fuelType - What type of fuel the car uses.
15. brand - The brand of the car.
16. notRepairedDamage - If the car has a damage which is not yet repaired.
17. dateCreated - The date on which the eBay listing was created.
18. nrOfPictures - The number of pictures in the ad.
19. postalCode - The postal code for the location of the vehicle.
20. lastSeenOnline - When the crawler saw this ad last online.

### Load Workspace

Data analysis in python usually begins by importing the relevant libraries. 
we've imported libraries for cleaning and manipulating tabular data, for visualizing data and for statistical analysis.

```python
import re
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy import stats
import statsmodels.formula.api as smf
```

### Data Cleaning

Our data cleaning process involved teh following actions:
* Renaming columns to snail case

```python
new_columns = {
    'dateCrawled':'date_crawled',  
    'offerType':'offer_type', 
    'price':'price_usd',
    'vehicleType':'vehicle_type', 
    'yearOfRegistration':'registration_year', 
    'powerPS':'power_ps', 
    'odometer':'odometer_km',
    'monthOfRegistration':'registration_month', 
    'fuelType':'fuel_type', 
    'notRepairedDamage':'unrepaired_damage', 
    'dateCreated':'date_created', 
    'nrOfPictures':'num_of_pictures', 
    'postalCode':'postal_code', 
    'lastSeen':'last_seen'
}

df = df.rename(columns=new_columns)
```

* Mapping values to convert German words to English

```python
df.assign(
    date_crawled=lambda x: pd.to_datetime(x.date_crawled),
    date_created=lambda x: pd.to_datetime(x.date_created),
    last_seen=lambda x: pd.to_datetime(x.last_seen),
    postal_code=lambda x: x.postal_code.astype(str),
    seller=lambda x: x.seller.map({'privat':'private', 'gewerblich':'commercial'}).fillna(x.seller),
    name=lambda x: x.name.str.replace('_', ' '),
    brand=lambda x: x.brand.str.replace('_', ' ').map({'sonstige autos':'other cars'}).fillna(x.brand),
    offer_type=lambda x: x.offer_type.map({'Angebot':'offer', 'Gesuch':'request'}).fillna(x.offer_type),
    price_usd=lambda x: x.price_usd.str.replace('$', '', regex=False).str.replace(',','', regex=False).astype(float),
    odometer_km=lambda x: x.odometer_km.str.replace('km', '', regex=False).str.replace(',','', regex=False).astype(float),
    gearbox=lambda x: x.gearbox.map({'manuell':'manual', 'automatik':'automatic'}).fillna(x.gearbox),
    vehicle_type=lambda x: x.vehicle_type.map({'kleinwagen':'small car', 'kombi':'station wagon', 'cabrio':'convertible', 'andere':'others'}).fillna(x.vehicle_type),
    fuel_type=lambda x: x.fuel_type.map({'benzin':'petrol', 'elektro':'electric', 'andere':'others'}).fillna(x.fuel_type),
    unrepaired_damage=lambda x: x.unrepaired_damage.map({'nein':'no', 'ja':'yes'}).fillna(x.unrepaired_damage),
    model=lambda x: x.model.map({'andere':'others'}).fillna(x.model),
)
```

* Converting data types to appropriate formats

* Dropping features with not enough variation in values.

```python
df = df.drop(columns=['num_of_pictures', 'seller', 'offer_type'])
```

* Replacing missing values in categorical features with the mode

```python
df.loc[df.vehicle_type.isna(), 'vehicle_type'] = df.vehicle_type.mode().iloc[0]
df.loc[df.gearbox.isna(), 'gearbox'] = df.gearbox.mode().iloc[0]
df.loc[df.fuel_type.isna(), 'fuel_type'] = df.fuel_type.mode().iloc[0]
df.loc[df.unrepaired_damage.isna(), 'unrepaired_damage'] = df.unrepaired_damage.mode().iloc[0]
```

* Correcting registration year for absurd values

```python
df.loc[df.registration_year < 1950, 'registration_year'] = 1950
```


### Exploratory & Statistical Analysis

**Distributions**

The numeric columns were distributed as follows:
![Numeric Distributions](LINK)

The categorical columns were distributed as follows:
![Categorical Distributions](LINK)


### Conclusion

For the distribution of numeric features:
* Price is very right skewed with 98.4% of values under 30k
* Registration year is left skewed, with most cars being registered after 1990.
* Horse power is severely right skewed as well.
* Odometer only has a few unique values, making a better categorical column than numeric. It is also left skewed, with most listings having odomoter values inexcess of 120km.

For distribution of Categorical features:
* More limousines are listed on the site
* Manual fearboxes are omre popular
* Petrol engines are more popular
* Cars with no unrepaired damaged are more popular.

In plotting price over time, there's no specific trend and there's no seaonality trend in price.

Nearly all listings are in the month of March.

Most listings stay onsite for less than 50 days. In determining why listings stay on site for longer, neither price, brand, vehicle type nor mileage is a good predictor.

The most expensive listed vehicle brand is the Porsche and the least expensive is the Daewoo.

The top 10 brands by avereage mileage is made up of luxury and non-luxury vehicle brand names.

Mercedex Benz is the most popular listing name.

The volkswagen golf is the most popular brand/model combination with nearly 4k listings. On average the car is listed for $4861.99.

When grouped by odometer, we gind that the higher the odometer value, the lower the average price. However, this hypothesis fails statistical testing.

None of the remaining variables are able to explain the variation in the price column, implying that there is no statistically significant relationship between price and other features. 

