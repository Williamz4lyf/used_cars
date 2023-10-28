# used_cars
# Analysis of German Used Car Sales from eBay Classifieds

The aim of this project is to clean the data and analyze the included used car listings. The listings are from Germany and some relevant information / values are in German. Our data cleaning process included making conversions from German to English. 

The data dictionary provided with data is as follows:

dateCrawled - When this ad was first crawled. All field-values are taken from this date.
name - Name of the car.
seller - Whether the seller is private or a dealer.
offerType - The type of listing
price - The price on the ad to sell the car.
abtest - Whether the listing is included in an A/B test.
vehicleType - The vehicle Type.
yearOfRegistration - The year in which the car was first registered.
gearbox - The transmission type.
powerPS - The power of the car in PS (horse power).
model - The car model name.
kilometer - How many kilometers the car has driven.
monthOfRegistration - The month in which the car was first registered.
fuelType - What type of fuel the car uses.
brand - The brand of the car.
notRepairedDamage - If the car has a damage which is not yet repaired.
dateCreated - The date on which the eBay listing was created.
nrOfPictures - The number of pictures in the ad.
postalCode - The postal code for the location of the vehicle.
lastSeenOnline - When the crawler saw this ad last online.

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
* Mapping values to convert German words to English
* Converting data types to appropriate formats
* Dropping features with not enough variation in values.
* Replacing missing values in categorical features with the mode
* Correcting registration year for absurd values


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

