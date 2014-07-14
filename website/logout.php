<?php

	session_start(); 

	session_unset();

	session_destroy();

	header('Location: http://ee-ug1.ee.ic.ac.uk/actual_web2/');
?>