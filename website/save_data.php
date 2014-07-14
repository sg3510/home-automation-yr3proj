<?php
$basicurl=sprintf('http://api.wunderground.com/api/96efddc47752583b/conditions/q/UK/London.json');

$json_reply = file_get_contents($basicurl);
$json=json_decode($json_reply);

$outside_temperature = $json->{'current_observation'}->{'temp_c'};
$outside_humidity = $json->{'current_observation'}->{'relative_humidity'};
$windspeed = $json->{'current_observation'}->{'wind_mph'};
$winddir = $json->{'current_observation'}->{'wind_degrees'};
$date = ($json->{'current_observation'}->{'local_epoch'}*1000);
$outside_humidity = substr($outside_humidity, 0,-1);

//$file = 'dump.txt';
// file_put_contents($file, $_POST);
// file_put_contents($file, $date);
echo $_POST;
echo " ";
echo $_POST['temperature_house'];
echo " ";
echo $_POST['inside_humidity'];

if ($outside_temperature!=NULL && $outside_humidity!=NULL && $windspeed!=NULL && $date!=NULL)
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
$sql="INSERT INTO `Data`.`Data` (`Outside_Temperature`,`Outside_Humidity`,`Wind_speed`,`Wind_direction`,`Inside_Temperature`,`Inside_Humidity`,`Time_stamp`) VALUES ('$outside_temperature','$outside_humidity','$windspeed','$winddir','$_POST[temperature_house]','$_POST[inside_humidity]','$date')";

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