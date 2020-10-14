Documentaion for Lungs AI:
	file Introduction:
		1. In lungs folder, four files related to the django(view.py,models.py,urls.py,lungs.html,pulseoxy.html)
		2. lungs.txt is the dataset used for the creation of ML file 
		3. pulseoxy_test.pkl is the Machine Learning AI code 

	Working Method:
		the html file are the front end part of the django, lungs.html is the initial page of the lungs meter.you can check that here "https://www.mydoctorspot.com/care/login_lite.php"
register and logon the page and click on Covid19 button then it load to page contain Lungs button.
when clickon that button you can see the "lungs.html"
After submiting the lungs.html it passes the dtas to the django view.py file. there lungs AI will run and pass the output to the pulseoxy.html page.
