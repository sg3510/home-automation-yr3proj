<?php
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","House_states");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }

for($j = 1; $j<=7; $j++)
{
	switch($j){
		case 1: $day="Monday";break;
		case 2: $day="Tuesday";break;
		case 3: $day="Wednesday";break;
		case 4: $day="Thursday";break;
		case 5: $day="Friday";break;
		case 6: $day="Saturday";break;
		case 7: $day="Sunday";break;
		}
	//mysqli_query($con,"CREATE TABLE $day (Hour TEXT, state TEXT)");	
	echo $day;
	for($i=0; $i<=23; $i++)
	{
		mysqli_query($con,"DELETE FROM $day WHERE Hour=$i");
		$x = rand(0,2);
		mysqli_query($con,"INSERT INTO $day VALUES ($i,$x)");
	}	
	
}



mysqli_close($con);

?>