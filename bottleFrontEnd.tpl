<html>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

	<head>
		<title>Opinion Mining Demo</title>
		<!--<link rel = Stylesheet type = "text/css" href = "/home/ubuntu/NewAWS_files/testcss.css">-->
		<style>
		
			h1 {
	text-align: center;
	font-size: 150%;
}

#hh{
	text-align: center;
	font-size: 100%;
}

* {
    box-sizing: border-box;
}

.col-1 {width: 8.33%;}
.col-2 {width: 16.66%;}
.col-3 {width: 25%;}
.col-4 {width: 33.33%;}
.col-5 {width: 41.66%;}
.col-6 {width: 50%;}
.col-7 {width: 58.33%;}
.col-8 {width: 66.66%;}
.col-9 {width: 75%;}
.col-10 {width: 83.33%;}
.col-11 {width: 91.66%;}
.col-12 {width: 100%;}

@media only screen and (max-width: 768px) {
    /* For mobile phones: */
    [class*="col-"] {
        width: 100%;
    }
}

[class*="col-"] {
    float: left;
    padding: 15px;
    border: 0px solid white;
}

.row::after {
    content: "";
    clear: both;
    display: table;
}

section{
	background-color: grey;
}

textarea{
	resize: none;
}

p {
	font-family: Arial, Helvetica, sans-serif;
	background-color: #ccc;
	display: block;
	font-size:2em;
	width: 100%
}

#textIn {
	width: 100%;
}

#fileIn{
	font-size: 20px;
	width:60%;
	padding:5px 5px 5px 5px;
	margin-top:15px;
	-webkit-border-radius: 5px;
    border-radius: 5px;
}

input[type=submit] {
	font-size: 20px;
    font-weight: 700;
    color: white;
    padding:5px 5px 5px 5px;
	margin-top:15px;
	width:100%;
    background:#ccc; 
    border:0 none;
    cursor:pointer;
    -webkit-border-radius: 5px;
    border-radius: 5px;
}

.subButton:hover{
	background-color: #cac;
}
/*input[type=file] {
    padding:15px 15px; 
    background:#ccc; 
    border:0 none;
    cursor:pointer;
    -webkit-border-radius: 5px;
    border-radius: 5px; 
	width: 100%;
}
*/
.inputfile {
	width: 0.1px;
	height: 0.1px;
	opacity: 0;
	overflow: hidden;
	position: absolute;
	z-index: -1;
}

.inputfile + label {
    font-size: 20px;
    font-weight: 700;
    color: white;
    display: inline-block;
	padding:5px 5px 5px 5px;
	margin-top:15px;
	width:40%;
    background:#ccc; 
    border:0 none;
    cursor:pointer;
    -webkit-border-radius: 5px;
    border-radius: 5px;
	float:right;
}

/*.inputfile:focus + label,*/
.inputfile + label:hover {
    background-color: #cac;
}
		
		</style>
	</head>

	<body>
		<section id = "banner"><h1>Opinion Mining Demo ~70% Accurate</h1></section>

		<section>
			<div class="row">
				<div class="col-8"> 
					<form method="post" action="/">
					
						<textarea id = "textIn" rows = "8" cols = "60" name='blog' placeholder = "Insert Blog Here"></textarea>
				</div>
				
				<div class="col-4">
                         
					<input type='submit' value='Submit'>
			
				</div>
					</form>
				
			</div>
			
			<p>{{message}}</p>
		</section>

	</body>

</html>
