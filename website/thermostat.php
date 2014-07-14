<?php session_start(); 

if($_SESSION['logedin'] == false)
  {
    header('Location: http://ee-ug1.ee.ic.ac.uk/actual_web2/');
  }

?>

<!DOCTYPE html>
<!--[if IE 8]>    <html class="no-js lt-ie9" lang="en"> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang="en"> <!--<![endif]-->
<head>
  <meta charset="utf-8" />
 <link rel="shortcut icon" href="http://ee-ug1.ee.ic.ac.uk/actual_web2/images/favicon.ico"/>
  <!-- Set the viewport width to device width for mobile -->
  <meta name="viewport" content="width=device-width" />

  <title>Welcome to Foundation</title>

  <link rel="stylesheet" href="css/normalize.css">
  </br>
  <link rel="stylesheet" href="css/foundation.css">

  <script src="js/vendor/custom.modernizr.js"></script>
  <script src="//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min.js"></script>

</head>
<body>


 <!-- Nav and Banner -->
  
  <div class="row">
    <div class="large-12 columns">
      <ul class="button-group">
        <li><a href="home.php" class="button" style="width: 200px">Home Page</a></li>
        <li><a href="control.php" class="button" style="width: 200px">House Control</a></li>
        <li><a href="stats.php" class="button" style="width: 200px">Statistics</a></li>
        <li><a href="thermostat.php" class="button" style="width: 200px">Thermostat Setting</a></li>
      </ul>
      
      <!-- Main Banner Image - place in a paragraph to enforce the modular scale spacing -->
      <p><img src="images/LogoTitle.png" /></p>
    </div>
  </div>
  
  <?php
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","Data");
$result = mysqli_query($con,"SELECT * FROM limits");
$rows = mysqli_fetch_array($result);
?>
 
 
  <!-- Main Page Content and Sidebar -->
 
  <div class="row">
 
    <!-- Contact Details -->
    <div class="nine columns" style="width: 800px">
 
      <h3>Manually Adjust Thermostat</h3>
      <p><font color="#ffffff">The temperatures displayed below are the current settings.</br></br>Please set your thermostat setting to the desired temperature below</em>.</p>  
	  
         <form action="" method="post" style="width: 400px" id="form">
         Thermostat Setting:<input type="text" name="User_Ts" value="Loading" id="aa"><br>
         <button type="submit" class="radius button">Submit</button>
         </form>     

		 <p>Set the limits of your thermostat below.</p>
		 
		<form action="" method="post" style="width: 400px" id="form2" >
		  Maximum Temperature: <input type="number" name="max" id="max" value=<?php echo $rows['Max']; ?>><br>
		  Minimum Temperature: <input type="number" name="min" id="min" value=<?php echo $rows['Min']; ?>><br>
		  Sleeping Temperature: <input type="number" name="sleep" id="sleep" value=<?php echo $rows['Sleep']; ?>><br>
		  <input type="submit"  class="radius button" value="Update">
		</form>
</font>
<?php mysqli_close($con); ?>

        </li>
      </ul>
 <footer class="row" style="width: 830px">
    <div class="large-12 columns" style="width:800px">
      <hr />
      <div class="row">
        <div class="large-6 columns" style="width:800px">
         <p style="text-align:right"><font color="#ffffff">&copy; Copyright @AutoHome</font>
           <a href="logout.php">Log out</a></p>
        </div>
      </div>
    </div> 
  </footer>
    </div>

  </div>
  
	<script type="text/javascript">

	var ws = new WebSocket("ws://ec2-54-214-164-65.us-west-2.compute.amazonaws.com:8888/ws");
	
	document.getElementById("max").min = document.getElementById("min").value;
	document.getElementById("min").max = document.getElementById("max").value;
	
	ws.onopen = function () {
				ws.send('{"type":"thermostat_request"}');
				};
	
	ws.onmessage = function (evt) {
			    //console.log("message: " + evt.data);
				msg_data = jQuery.parseJSON(evt.data);
				
				if(msg_data.type=="thermostat_request_response")
				{
					document.getElementById("aa").value = msg_data.temperature;
					document.getElementById("aa").type = "number";
					document.getElementById("aa").min = document.getElementById("min").value;
					document.getElementById("aa").max = document.getElementById("max").value;
				}
				};
	
    $("#form").submit(function(e){
		e.preventDefault();
        var str = $("#form").serialize();
		
		if(document.getElementById("aa").type == "text")
		{
			alert("Please let us load the current setting first!");
		}
		else if(str.replace("User_Ts=","") == "")
		{
			alert("Please enter a number!");
		}
		else if(str.replace("User_Ts=","") < document.getElementById("min").value || str.replace("User_Ts=","") > document.getElementById("max").value)
		{
			alert("Please eneter a temperature betweem the set limits!");
		}
		else
		{
			$.post('submit.php', str, function(result){
				alert("Thermostat Updated!");
				ws.send('{"type":"thermostat_set", "temperature": "'+str.replace("User_Ts=","")+'"}');
			})  
		}
    });
	
	$("#form2").submit(function(e){
		e.preventDefault();
        var str = $("#form2").serialize();
		document.getElementById("max").min = document.getElementById("min").value;
		document.getElementById("min").max = document.getElementById("max").value;
		document.getElementById("aa").min = document.getElementById("min").value;
		document.getElementById("aa").max = document.getElementById("max").value;
        $.post('submittemp.php', str, function(result){
            alert(result);
        })        
    });
</script>
 
  <!-- End Main Content and Sidebar -->
 

 
  <!-- Footer -->
  

          
</body>
</html>