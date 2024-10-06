/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you: 
you might need to do some digging, aand revise the Working with Relational Databases in Python chapter in the previous resource.

Otherwise, the questions in the case study are exactly the same as with Tier 1. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */
SELECT name
FROM Facilities
WHERE membercost != 0;

/* Q2: How many facilities do not charge a fee to members? */
SELECT COUNT(name)
FROM Facilities
WHERE membercost = 0;

/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT 
    facid, 
    name, 
    membercost, 
    monthlymaintenance
FROM  Facilities
WHERE membercost !=0
    AND membercost < 0.2 * monthlymaintenance;

/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */
SELECT *
FROM Facilities
WHERE facid IN ( 1, 5 )

/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */
SELECT 
    name, 
    monthlymaintenance,
    CASE WHEN monthlymaintenance >100
            THEN  'expensive'
        ELSE  'cheap'
        END AS label
FROM Facilities;

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */
SELECT 
	firstname,
	surname
FROM Members 
WHERE joindate =
	(SELECT MAX(joindate)
     FROM Members);

/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT 
	DISTINCT f.name AS facility_name,
	CONCAT_WS(' ', m.firstname, m.surname) AS member_name
FROM Facilities AS f
INNER JOIN Bookings AS b
ON b.facid = f.facid
INNER JOIN Members AS m
ON b.memid = m.memid
WHERE LOWER(f.name) LIKE ('%tennis court%')
ORDER BY member_name;



/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT 
	f.name AS facility_name,
	CONCAT_WS(' ', m.firstname, m.surname) AS name,
	CASE WHEN b.memid = 0 THEN f.guestcost * b.slots
		ELSE f.membercost * b.slots
	END AS cost
FROM Bookings AS b
INNER JOIN Facilities AS f
ON f.facid = b.facid
INNER JOIN Members AS m
ON m.memid = b.memid
WHERE DATE(b.starttime) = "2012-09-14" 
	AND ((b.memid = 0 AND f.guestcost * b.slots > 30) 
     OR (b.memid <> 0 AND f.membercost * b.slots > 30))
ORDER BY cost DESC;



/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT
	facility_name,
	personal_name, 
	rate * slots AS cost
FROM (SELECT
		f.name AS facility_name,
      	CONCAT_WS(' ', m.firstname, m.surname) AS personal_name,
    	CASE WHEN b.memid = 0 THEN f.guestcost
    		ELSE f.membercost END AS rate,
		b.slots
    FROM (
        SELECT *
        FROM Bookings
        WHERE DATE(starttime) = '2012-09-14') AS b
    LEFT JOIN Facilities AS f
    ON f.facid = b.facid
    LEFT JOIN Members AS m
    ON m.memid = b.memid) AS bm
WHERE bm.rate * bm.slots > 30
ORDER BY cost DESC;


/*!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!
VERSION TWO
!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!
*/
SELECT *
FROM (
	SELECT
    	f.name AS facility_name,
		(SELECT CONCAT_WS(' ', firstname, surname)
    	 FROM Members AS m
	     WHERE m.memid = b.memid) AS personal_name,
		(b.slots * (SELECT 
			CASE WHEN b.memid = 0 THEN f.guestcost
    			ELSE f.membercost END
	     FROM Facilities AS f
    	 WHERE f.facid = b.facid)) AS cost
	FROM Bookings AS b
	LEFT JOIN Facilities AS f
	ON f.facid = b.facid
	WHERE DATE(b.starttime) = '2012-09-14') AS res
WHERE res.cost > 30
ORDER BY res.cost DESC;



/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook 
for the following questions.  

!!!!!!!!! My answer
!!!!!!!!!
Unfortunately, I couldn’t find a direct way to export the country club data from PHPMyAdmin into SQLite. 
As a result, I followed these steps:
	1.	Downloaded each table from PHPMyAdmin in .json format (3 files, named after each table).
	2.	Removed all comments from the downloaded files.
	3.	Developed a Python script to create a SQLite database with 3 tables (Bookings, Facilities, and Members) 
	and to insert data from the downloaded .json files, using the Pandas DataFrame method to_sql.

Here is my Python script:

import sqlite3
import pandas as pd
import json

connection = sqlite3.connect("country_club_SQLite.db")
cursor = connection.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS 'Bookings' \
    ('bookid'   INTEGER, \
    'facid'     INTEGER, \
    'memid'     INTEGER, \
    'starttime' TEXT, \
    'slots'     INTEGER);")

cursor.execute("CREATE TABLE IF NOT EXISTS 'Members' \
    ('memid'    INTEGER, \
    'surname'   TEXT, \
    'firstname' TEXT, \
    'address'   TEXT, \
    'zipcode'   INTEGER, \
    'telephone' TEXT, \
    'recommendedby' TEXT, \
    'joindate'  TEXT);")

cursor.execute("CREATE TABLE IF NOT EXISTS 'Facilities' \
  ('facid' INTEGER, \
  'name' TEXT, \
  'membercost' NUMERIC, \
  'guestcost' NUMERIC, \
  'initialoutlay' INTEGER, \
  'monthlymaintenance' INTEGER);")

connection.commit()

tables = ['Bookings', 'Facilities', 'Members']

for table in tables:
    with open(table + ".json") as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df.to_sql(table, connection, if_exists='append', index=False)
    connection.commit()

*/

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */


/* Q12: Find the facilities with their usage by member, but not guests */


/* Q13: Find the facilities usage by month, but not guests */

