<?php
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");

mysqli_query($con,"INSERT INTO test (`test`) VALUES ('$_POST[data]')");
mysqli_close($con);

echo "You sent: ". $_POST['data'];
?>