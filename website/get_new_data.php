<?php
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }

$result = mysqli_query($con,"SELECT * FROM Data2 ORDER BY Time_stamp");
echo '"([';
while($row = mysqli_fetch_array($result))
  {
  echo '[' . $row['Time_stamp'] . ',' . $row['Temperature'] . '],';

  }

 echo ']);"';

mysqli_close($con);
  

?>