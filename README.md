SuPave Paving Temperature Analysis App Manual
This tool allows users to analyze temperature data collected during asphalt paving operations. It identifies cold zones, calculates thermal segregation, stop durations, and visualizes temperature profiles spatially and statistically.
 
1.	 Required Input Format
Upload an .xlsx file that matches the following structure (as in Pave_temp_template.xlsx):
Required Columns:


Time: Timestamp in format dd/mm/yyyy hh:mm:ss UTC +0000
Moving distance:	Paver's cumulative distance in meters
Latitude:	GPS coordinate
Longitude:	GPS  coordinate
Easting:	GPS coordinate
Northing:	GPS  coordinate
Numeric widths:	Contains one column per paving width (eg 6.5,6.25,….,-6.5) with temperatures in °C


2.	 App Inputs
In the sidebar, you can control:
•	Paving Width Threshold (m):
Select the min/max range of paving width to analyze (e.g., -2.0 to 2.0 m).
•	Show Cold Spots (<120°C):
Highlight areas considered too cold for proper compaction.
•	Show Risk Areas (<90% avg temp):
Identify zones at risk of segregation.
3.	Tool output
Temperature Heatmap with Stop Lines
•	Visualizes temperature across paving width and distance.
•	Reports total idle time duration.

Cold Spots Map: Highlights regions <120°C over the paving area.
Risk Spot Map: Marks zones below 90% of the overall average temperature.
Thermal Segregation Index (TSI)
•	Shows average temperature variation across the mat.
•	Classifies severity: Low, Moderate, or High.

Differential Range Statistics (DRS)
•	Calculates T10, T90, DRS (°C).
•	Includes severity class and a KDE-enhanced histogram.

Temperature Trends
•	Plot of average row temperature vs. moving distance using Plotly.
GPS Track Map
•	Interactive map using PyDeck with tooltips.
•	Shows paver movement path and data points.

4.	 Summary & Export
•	Summary text includes key statistics and input parameters.
•	You can download the full summary as a .txt report.

Troubleshooting
•	File not uploading? Ensure it's .xlsx and follows the template format.
•	Missing plot? Check if GPS data or width columns are missing or zero-filled.
•	Crashes on load? Confirm column headers exactly match expected names (case-sensitive).

