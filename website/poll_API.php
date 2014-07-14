<?php
//~ http://api.worldweatheronline.com/free/v1/weather.ashx?key=xxxxxxxxxxxxxxxxx&q=SW1&num_of_days=3&format=json

//Minimum request
//Can be city,state,country, zip/postal code, IP address, longtitude/latitude. If long/lat are 2 elements, they will be assembled. IP address is one element.
$loc_array= Array("London");		//data validated in foreach. 
$api_key="s7nqncjrk4y44et3dsmr2aw8";		//should be embedded in your code, so no data validation necessary, otherwise if(strlen($api_key)!=24)
$num_of_days=0;					//data validated in sprintf

$loc_safe=Array();
foreach($loc_array as $loc){
	$loc_safe[]= urlencode($loc);
}
$loc_string=implode(",", $loc_safe);

//To add more conditions to the query, just lengthen the url string
// $basicurl=sprintf('http://api.worldweatheronline.com/free/v1/weather.ashx?key=%s&q=%s&num_of_days=%s&format=json', 
	// $api_key, $loc_string, intval($num_of_days));
$basicurl=sprintf('http://api.wunderground.com/api/96efddc47752583b/conditions/q/UK/London.json');

$json_reply = file_get_contents($basicurl);
$json=json_decode($json_reply);

// $temperature = $json->{'data'}->{'current_condition'}['0']->{'temp_C'};
// $humidity = $json->{'data'}->{'current_condition'}['0']->{'humidity'};
// $windspeed = $json->{'data'}->{'current_condition'}['0']->{'windspeedMiles'};
// $winddir = $json->{'data'}->{'current_condition'}['0']->{'winddirDegree'};
$temperature = $json->{'current_observation'}->{'temp_c'};
$humidity = $json->{'current_observation'}->{'relative_humidity'};
$windspeed = $json->{'current_observation'}->{'wind_mph'};
$winddir = $json->{'current_observation'}->{'wind_degrees'};
$date = ($json->{'current_observation'}->{'local_epoch'}*1000);
$humidity = substr($humidity, 0,-1);

echo $temperature;
echo " ";
echo $humidity;
echo " ";
echo $windspeed;
echo " ";
echo $winddir;
echo " ";


if ($temperature!=NULL && $humidity!=NULL && $windspeed!=NULL && $date!=NULL)
{
echo 'yup';
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }
//$date = date ('Y,m,d,H,i,s');
//$date = (time() * 1000);
$sql="INSERT INTO `Data`.`Data2` (`Temperature`, `Humidity`, `Wind_speed`, `Wind_direction`, `Time_stamp`) VALUES ('$temperature','$humidity','$windspeed','$winddir', '$date')";

if (!mysqli_query($con,$sql))
  {
  die('Error: ' . mysqli_error($con));
  }
echo "1 record added";

mysqli_close($con);
}
else
{
echo "nope";
}

?>