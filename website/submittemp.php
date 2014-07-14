<?php

$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");
// Check connection
if (mysqli_connect_errno())
 {
	echo "Failed to connect to MySQL: " . mysqli_connect_error();
 }

if($_POST[max]=="" || $_POST[min]=="" || $_POST[sleep]=="")
{
	echo "Please ensure all temperatures are filled in";
}
elseif($_POST[max] <= $_POST[min])
{
	echo "Max Temperature must be higher than Min Temperature!";
}
else
{ 
	mysqli_query($con,"DELETE FROM limits"); 
	  
	$sql="INSERT INTO limits (`Max`, `Min`, `Sleep`) VALUES ('$_POST[max]','$_POST[min]','$_POST[sleep]')";

	if (!mysqli_query($con,$sql))
	  {
	  die('Error: ' . mysqli_error($con));
	  }
	echo "Temperatures Updated!";
}

mysqli_close($con);


?>