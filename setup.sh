mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"sdupland@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
base=\"dark\"\n\
primaryColor=\"#4be8e0\"\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
