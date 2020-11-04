from scraping import AutoRuData

url = 'https://auto.ru/moskovskaya_oblast/cars/bmw/all/'
path = 'data.csv'

auto = AutoRuData(url)
autp.read_data()
auto.write_csv()
