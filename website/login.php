<?php

	session_start(); 

	$login = false;

	$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Users");

	$result = mysqli_query($con,"SELECT * FROM user_list");

	while($row = mysqli_fetch_array($result))
 	{
 	 if($row['username']==$_POST['username'] && $row['password']==$_POST['password'])
 	 {
 	 	$login = true;
 	 }
	}

	if ($login == false)
	{
		echo "False";
	}
	else
	{
		$_SESSION['logedin'] = true;
		$_SESSION['user'] = $_POST['username'];
		echo "True";
	}

?>