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

  <title>Welcome to your AutoHome!</title>

  <link rel="stylesheet" href="css/normalize.css">
  </br>
  <link rel="stylesheet" href="css/foundation.css">

  <script src="custom.modernizr.js"></script>
  
  <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
  <script src="http://code.highcharts.com/stock/highstock.js"></script>
  <script src="http://code.highcharts.com/stock/modules/exporting.js"></script>

</head>
<body>

<!-- Nav and Banner -->
  
  <div class="row" style="margin: 0 auto">
    <div class="large-12 columns">
      <ul class="button-group">
        <li><a href="home.php" class="button" style="width: 200px">Home Page</a></li>
        <li><a href="control.php" class="button" style="width: 200px">House Control</a></li>
        <li><a href="stats.php" class="button" style="width: 200px">Statistics</a></li>
        <li><a href="thermostat.php" class="button" style="width: 200px">Thermostat Setting</a></li>
      </ul>
      
      <!-- Main Banner Image - place in a paragraph to enforce the modular scale spacing -->
      <p><img src="images/LogoTitle.png" /></p>
	  
	  <p>  
<div id="chart1" style="height: 500px; width: 800px"></div>
</br>
<div id="chart2" style="height: 500px; width: 800px"></div>
</p>




 </br>
<?php
$con=mysqli_connect("127.0.0.1","mike","ChickenKorma","House_states");
// Check connection
if (mysqli_connect_errno())
  {
  echo "Failed to connect to MySQL: " . mysqli_connect_error();
  }

echo '<table>';
echo '<tr>';
echo '<td>Day</td>';

for ($i = 0; $i<=23; $i++)
{
	echo '<td>'.$i.'</td>';
}
echo '</tr>';

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
	
	$result = mysqli_query($con,"SELECT * FROM $day ORDER BY Hour");
	
	echo '<tr>';
	echo '<td>'.$day.'</td>';
	
	while($row = mysqli_fetch_array($result))
	{		
	
		if($row['state'] == 6 || $row['state'] == 0 || $row['state'] == 1 || $row['state'] == 2)
		{			
			$var = mysqli_fetch_array(mysqli_query($con,"SELECT state FROM Auto_$day WHERE Hour='$row[Hour]'"));
			$row['state'] = $var['state'];
			mysqli_query($con,"UPDATE $day SET state='$row[state]' WHERE Hour=$row[Hour]");
		}
		
		echo '<td id="'.$day.$row['Hour'].'" onclick="showForm(this.id)" ';
		if($row['state'] == 0 || $row['state'] == 3)
		{
			echo 'title="Asleep" style="background-color:red; ';
		}
		elseif($row['state'] == 1 || $row['state'] == 4)
		{	
			echo 'title="In" style="background-color:green; ';
		}
		elseif($row['state'] == 2 || $row['state'] == 5)
		{	
			echo 'title="Out" style="background-color:blue; ';
		}
		if($row['state'] == 3 || $row['state'] == 4 || $row['state'] == 5)
		{
			echo 'opacity:0.8"';
		}
		else
		{
			echo 'opacity:1">';
		}
		// echo '">';
		echo '</td>';
	}
	echo '</tr>';
}
  
  echo '</tr>';
  echo '</table>';
  



mysqli_close($con);

?>

<div id="updatestate" style="display: none">
	<font color="#ffffff">Mark the selected cells as: <div id="location" style="display:none"></div></font>
	<select style="width:200px">
		<option value="Asleep">Asleep</option>
		<option value="In">In</option>
		<option value="Out">Out</option>
		<option value="Auto">Auto</option>
	</select>
	
	<input type="button" onClick="updateCell($('#updatestate option:selected').text())" class="radius button" value="Update">
	<input type="button" onClick="unselectall()" class="radius button" value="Unselect All">
	
</div>
<input type="button" onClick="clearAll()" class="radius button" value="Reset All To Auto">



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
  <!-- End Nav and Banner -->
  

  
  <!-- Footer -->
  
  
  
<script type='text/javascript'>
	function showForm(cell)
	{	
		var cell2 = eval(cell);
		if(document.getElementById("updatestate").style.display !== "none")
		{
			if(document.getElementById("location").innerHTML.indexOf(cell+',') == -1)
			{
				document.getElementById("location").innerHTML += cell + ', ';
				//cell2.style.border = "medium solid";
				cell2.style.opacity *= 0.01;
			}
			else
			{
				document.getElementById("location").innerHTML = document.getElementById("location").innerHTML.replace(cell+', ', "");
				//cell2.style.border = "none";
				cell2.style.opacity *= 100;
			}
		}
		else
		{
			//document.getElementById("updatestate").style.display = "inherit";
			$('#updatestate').slideDown();
			document.getElementById("location").innerHTML = cell+', ';
			//cell2.style.border = "medium solid";
			cell2.style.opacity *= 0.01;
		}
		
		if(document.getElementById("location").innerHTML.trim() == "")
		{
			//document.getElementById("updatestate").style.display = "none";
			$('#updatestate').slideUp();
		}
	}
	
	function updateCell(state)
	{
		//alert(state);
		var y = document.getElementById("location").innerText;

		var cells = y.split(",");
		
		for(var i=0; i<cells.length-1; i++)
		{		
			var x = cells[i];
			str = x.match(/([A-z]+)(\d{1,2})/);
			var newstate;
			x=eval(x);			
			x.style.opacity = 0.8;
			x.style.border = "none";
			
			if(state == "Asleep")
			{
				x.style.backgroundColor = "red";
				x.title="Asleep";
				newstate = 3;
			}
			else if(state == "In")
			{	
				x.style.backgroundColor = "green";
				x.title="In";
				newstate = 4;
			}
			else if(state == "Out")
			{	
				x.style.backgroundColor = "blue";
				x.title="Out";
				newstate = 5;
			}	
			else if(state == "Auto")
			{	
				x.style.backgroundColor = "orange";
				x.title="Please refresh to update";
				newstate = 6;
				x.style.opacity = 1;
			}			
			$.post('cellUpdate.php', {'data[]': str, 'state': newstate}, function(data){
				//console.log(data);
				});
		}
		$('#updatestate').slideUp();
	}
	
	function clearAll()
	{
		var day;
		var ALL = "";
		
		for(var i = 1; i<=7; i++)
		{
			switch(i){
				case 1: day="Monday";break;
				case 2: day="Tuesday";break;
				case 3: day="Wednesday";break;
				case 4: day="Thursday";break;
				case 5: day="Friday";break;
				case 6: day="Saturday";break;
				case 7: day="Sunday";break;
			}
			
			for(var j=0; j<=23; j++)
			{
				var temp = eval(day+j);

				ALL = ALL+day+j+", ";		
			}
		}
		$('#updatestate').slideUp();
		document.getElementById("location").innerText = ALL;
		updateCell("Auto");
		setTimeout(function(){document.location.reload(true)},500);
	}
	
	function unselectall(state)
	{
		//alert(state);
		var y = document.getElementById("location").innerText;
		
		var cells = y.split(",");
		
		for(var i=0; i<cells.length-1; i++)
		{	
			var x = cells[i];
			x = eval(x);
			x.style.opacity *= 100;
		}
		document.getElementById("location").innerText = "";
		$('#updatestate').slideUp();
	}

  $(function() {

	$.getJSON('http://ee-ug1.ee.ic.ac.uk/actual_web2/get_new_data.php', function(data) {
		// Create the chart	
		var b = eval(data);
		$('#chart1').highcharts('StockChart', {
				
			rangeSelector : {
				selected : 1,
				buttons: [{
					type: 'day',
					count: 1,
					text: '1d'
				}, {
					type: 'week',
					count: 1,
					text: '1w'
				}, {
					type: 'month',
					count: 1,
					text: '1m'
				}, {
					type: 'month',
					count: 6,
					text: '6m'
				}, {
					type: 'year',
					count: 1,
					text: '1y'
				}, {
					type: 'all',
					text: 'All'
				}]				
			},		
			

			title : {
				text : 'Temperature'
			},
			
			xAxis: {
				ordinal: false 
			},
			
			yAxis: [{
		        title: {
		            text: 'Temperature'
		        }
		    }, {
		        title: {
		            text: 'Humidity'
		        },
		        opposite: true
		    }],
			
			series : [{
				name : 'Temperature',
				data : b,
				tooltip: {
					valueDecimals: 0,
					valueSuffix: '°C'
				}
			}]
		});
		$.getJSON('http://ee-ug1.ee.ic.ac.uk/actual_web2/get_new_data2.php', function(data2) {
		var c = eval(data2);
		var chart = $('#chart1').highcharts();
		chart.addSeries({
				name : 'Humidity',
				data : c,
				yAxis: 1,
				tooltip: {
					valueDecimals: 0,
					valueSuffix: '%'
				}
                            });
		});
	});

});


  $(function() {

	$.getJSON('http://ee-ug1.ee.ic.ac.uk/actual_web2/get_new_data3.php', function(data) {
		// Create the chart	
		var b = eval(data);
		$('#chart2').highcharts('StockChart', {
			
			rangeSelector : {
				selected : 1,
				buttons: [{
					type: 'day',
					count: 1,
					text: '1d'
				}, {
					type: 'week',
					count: 1,
					text: '1w'
				}, {
					type: 'month',
					count: 1,
					text: '1m'
				}, {
					type: 'month',
					count: 6,
					text: '6m'
				}, {
					type: 'year',
					count: 1,
					text: '1y'
				}, {
					type: 'all',
					text: 'All'
				}]				
			},

			title : {
				text : 'Wind'
			},
			
			xAxis: {
				ordinal: false 
			},
			
			yAxis: [{
		        title: {
		            text: 'Wind Speed'
		        }
		    }, {
		        title: {
		            text: 'Wind Direction'
		        },
		        opposite: true
		    }],
			
			series : [{
				name : 'Wind Speed',
				data : b,
				tooltip: {
					valueDecimals: 0,
					valueSuffix: 'mph'
				}
			}]
		});
		$.getJSON('http://ee-ug1.ee.ic.ac.uk/actual_web2/get_new_data4.php', function(data2) {
		var c = eval(data2);
		var chart = $('#chart2').highcharts();
		chart.addSeries({
				name : 'Wind Direction',
				data : c,
				yAxis: 1,
				tooltip: {
					valueDecimals: 0,
					valueSuffix: 'degrees'
				}
                            });
		});
	});

});


</script>
 

</body>
</html>