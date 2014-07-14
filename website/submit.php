<?php

$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }
 $result =  mysqli_query($con,"SELECT * FROM Data2 ORDER BY Time_stamp DESC");
 $row = mysqli_fetch_array($result); 
$temperature = $row['Temperature'];
$humidity = $row['Humidity'];
$windspeed = $row['Wind_speed'];
$winddir = $row['Wind_direction'];

$date = time();

$sql="INSERT INTO `Data`.`user_update` (`User_Ts`, `Temperature`, `Humidity`, `Wind_speed`, `Wind_direction`, `Time_stamp`) VALUES ('$_POST[User_Ts]','$temperature','$humidity','$windspeed','$winddir', '$date')";

if (!mysqli_query($con,$sql))
  {
  die('Error: ' . mysqli_error($con));
  }
echo "1 record added";

mysqli_close($con);


?>
