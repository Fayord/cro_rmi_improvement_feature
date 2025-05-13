# Plot Network Visualization

This directory contains scripts and resources for visualizing risk-related networks using Dash and Cytoscape.

## How to Run

1. Ensure you have Python 3 and all required dependencies installed.
2. To start the Dash app, run the following shell script:

    ```bash
    ./run_plot_network.sh
    ```

3. Open your browser and go to [http://localhost:7070](http://localhost:7070) to view the app.

---

## How to add new url to vision vm
1. go to folder /etc/nginx/sites-available/ then create a new file or edit /etc/nginx/sites-available/dashly
1. add the following code:
    ```bash
    # {COMMENT_SECTION}
    location /{YOUR_APPLICATION_NAME} {
        proxy_pass http://localhost:{YOUR_APPLICATION_PORT};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    ```
    {COMMENT_SECTION}: This is a comment section. You can add any comments you want to explain the purpose of this section.
    {YOUR_APPLICATION_NAME}: Replace this with the name of your application.
    {YOUR_APPLICATION_PORT}: Replace this with the port number where your application is running.
1. check is the file is correct
    to check the file is correct, run the following command:
    ```bash
    sudo nginx -t
    ```
    if the file is correct, you will see the following message:
    ```bash
    nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
    nginx: configuration file /etc/nginx/nginx.conf test is successful
    ```
    if the file is not correct, you will see the following message:
    ```bash
    nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
    nginx: configuration file /etc/nginx/nginx.conf test is failed
    ```
1. restart nginx or start nginx
    to restart nginx, run the following command:
    ```bash
    sudo systemctl restart nginx
    ```

    to start nginx, run the following command:
    ```bash
    sudo systemctl start nginx
    ```

1. check the status of nginx
    to check the status of nginx, run the following command:
    ```bash
    sudo systemctl status nginx
    ```