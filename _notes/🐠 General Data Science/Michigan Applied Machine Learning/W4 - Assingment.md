---
---

# W4 Assignment

Predict whether a given blight ticked will be paid in one time.

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.

All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:

* [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
* [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
* [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
* [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
* [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)

## The datasets
- Target variable: `compliance`
	- `early`, `on time`, `within one month`, `False`, `Null`
		- Tickets with `Null` will not be considered inside the test set: They represent the case when the violator was not responsible of the lack of the payment

**File descriptions** (Use only this data for training your model!)

    train.csv - the training set (all tickets issued 2004-2011)
    test.csv - the test set (all tickets issued 2012-2016)
    addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
     Note: misspelled addresses may be incorrectly geolocated.

<br>

**Data fields**

train.csv & test.csv

- ticket_id - unique identifier for tickets
- agency_name - Agency that issued the ticket
- inspector_name - Name of inspector that issued the ticket
- violator_name - Name of the person/organization that the ticket was issued to
- violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
- mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
- ticket_issued_date - Date and time the ticket was issued
- hearing_date - Date and time the violator's hearing was scheduled
- violation_code, violation_description - Type of violation
- disposition - Judgment and judgement type
- fine_amount - Violation fine amount, excluding fees
- admin_fee - $20 fee assigned to responsible judgments
- state_fee - $10 fee assigned to responsible judgments
    late_fee - 10% fee assigned to responsible judgments
    discount_amount - discount applied, if any
    clean_up_cost - DPW clean-up or graffiti removal cost
    judgment_amount - Sum of all fines and fees
    grafitti_status - Flag for graffiti violations
    
train.csv only

- payment_amount - Amount paid, if any
- payment_date - Date payment was made, if it was received
- payment_status - Current payment status as of Feb 1 2017
- balance_due - Fines and fees still owed
- collection_status - Flag for payments in collections
- compliance [target variable for prediction] 
-  Null = Not responsible
-  0 = Responsible, non-compliant
-  1 = Responsible, compliant
- compliance_detail - More information on why each ticket was marked compliant or non-compliant
