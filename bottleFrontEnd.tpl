<html>
  <head>
      <title>Form Example</title>
  </head>
  <body>
    <form method="post" action="/">
        <fieldset>
            <legend>Blog sentiment tester</legend>
            <ul>
                <li>Insert Blog: <input name='blog'>
                </li> 
            </ul><input type='submit' value='Submit Form'>
        </fieldset>
    </form>
    
    <p>{{message}}</p>

  </body>
</html>