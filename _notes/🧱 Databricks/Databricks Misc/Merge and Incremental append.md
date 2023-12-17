
# Table Deletes, updates, and merges
<div class="rich-link-card-container"><a class="rich-link-card" href="https://docs.delta.io/latest/delta-update.html#operation-semantics&language-python" target="_blank">
	<div class="rich-link-image-container">
		<div class="rich-link-image" style="background-image: url('https://docs.delta.io/latest/_static/favicon.ico')">
	</div>
	</div>
	<div class="rich-link-card-text">
		<h1 class="rich-link-card-title">Table deletes, updates, and merges — Delta Lake Documentation</h1>
		<p class="rich-link-card-description">
		Learn how to delete data from and update data in Delta tables.
		</p>
		<p class="rich-link-href">
		https://docs.delta.io/latest/delta-update.html#operation-semantics&language-python
		</p>
	</div>
</a></div>

## Merge

## Upsert into a table using merge

You can upsert data from a source table, view, or DataFrame into a target Delta table by using the `MERGE` SQL operation. Delta Lake supports inserts, updates and deletes in `MERGE`, and it supports extended syntax beyond the SQL standards to facilitate advanced use cases.

Suppose you have a source table named `people10mupdates` or a source path at `/tmp/delta/people-10m-updates` that contains new data for a target table named `people10m` or a target path at `/tmp/delta/people-10m`. Some of these new records may already be present in the target data. To merge the new data, you want to update rows where the person’s `id` is already present and insert the new rows where no matching `id` is present. You can run the following:

```python
from delta.tables import *

deltaTablePeople = DeltaTable.forPath(spark, '/tmp/delta/people-10m')
deltaTablePeopleUpdates = DeltaTable.forPath(spark, '/tmp/delta/people-10m-updates')

dfUpdates = deltaTablePeopleUpdates.toDF()

deltaTablePeople.alias('people') \
  .merge(
    dfUpdates.alias('updates'),
    'people.id = updates.id'
  ) \
  .whenMatchedUpdate(set =
    {
      "id": "updates.id",
      "firstName": "updates.firstName",
      "middleName": "updates.middleName",
      "lastName": "updates.lastName",
      "gender": "updates.gender",
      "birthDate": "updates.birthDate",
      "ssn": "updates.ssn",
      "salary": "updates.salary"
    }
  ) \
  .whenNotMatchedInsert(values =
    {
      "id": "updates.id",
      "firstName": "updates.firstName",
      "middleName": "updates.middleName",
      "lastName": "updates.lastName",
      "gender": "updates.gender",
      "birthDate": "updates.birthDate",
      "ssn": "updates.ssn",
      "salary": "updates.salary"
    }
  ) \
  .execute()
```

```sql
MERGE INTO people10m
USING people10mupdates
ON people10m.id = people10mupdates.id
WHEN MATCHED THEN
  UPDATE SET
    id = people10mupdates.id,
    firstName = people10mupdates.firstName,
    middleName = people10mupdates.middleName,
    lastName = people10mupdates.lastName,
    gender = people10mupdates.gender,
    birthDate = people10mupdates.birthDate,
    ssn = people10mupdates.ssn,
    salary = people10mupdates.salary
WHEN NOT MATCHED
  THEN INSERT (
    id,
    firstName,
    middleName,
    lastName,
    gender,
    birthDate,
    ssn,
    salary
  )
  VALUES (
    people10mupdates.id,
    people10mupdates.firstName,
    people10mupdates.middleName,
    people10mupdates.lastName,
    people10mupdates.gender,
    people10mupdates.birthDate,
    people10mupdates.ssn,
    people10mupdates.salary
  )
```