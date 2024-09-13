const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public'))); 

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/submit', (req, res) => {
    const formData = req.body;
    const dataRow = Object.values(formData).join(',') + '\n';

    fs.appendFile('data_points.csv', dataRow, (err) => {
        if (err) {
            console.error('Error writing to CSV file', err);
            return res.status(500).send('Error saving data');
        }
        res.send('Data successfully saved to CSV!');
    });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
