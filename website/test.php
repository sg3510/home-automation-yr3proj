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
	
	$result = mysqli_query($con,"SELECT * FROM Auto_$day ORDER BY Hour");
	

	
	while($row = mysqli_fetch_array($result))
	{		
	
		if($row['state'] == 5)
		{			
			//$var = mysqli_fetch_array(mysqli_query($con,"SELECT state FROM Auto_$day WHERE Hour='$row[Hour]'"));
			$row['state'] = 2;
			echo $row['state'];
			mysqli_query($con,"UPDATE Auto_$day SET state='$row[state]' WHERE Hour=$row[Hour]");
			
			
		}
		
		
		
	
	}

}
  

  



mysqli_close($con);

?>