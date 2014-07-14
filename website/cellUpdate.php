<?php
		$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","House_states");
		// Check connection
		if (mysqli_connect_errno())
		{
			echo "Failed to connect to MySQL: " . mysqli_connect_error();
		}
		
		$var = $_POST[data];
		
		// echo $var[0];
		// echo $_POST[state];
		
		mysqli_query($con,"UPDATE $var[1] SET state=$_POST[state] WHERE Hour=$var[2]");

		echo $var[1].$var[2]." updated to ".$_POST[state];
		
		mysqli_close($con);
		
?>