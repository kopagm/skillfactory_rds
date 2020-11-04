import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm


class AutoRuData:

    CLASS_NEXT = 'Button Button_color_white Button_size_s Button_type_link Button_width_default ListingPagination-module__next'

    def __init__(self, url):
        self.url = url
        self.links = None
        self.df = None

    def reset_data(self):
        self.links = None
        self.df = None

    def get_soup(self, url):
        request = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        request.encoding = 'utf-8'
        return BeautifulSoup(request.text, 'html.parser')

    def get_links(self):
        if self.links is None:
            self.read_links()
        return self.links

    def set_links(self, links):
        self.reset_data()
        self.links = links

    def read_links(self):
        soup = self.get_soup(self.url)
        soup_next = soup.find('a', class_=self.CLASS_NEXT)

        s_links = soup.find_all(
            'a', class_='Link ListingItemTitle-module__link')
        links = [link.get('href') for link in s]
        while soup_next is not None:
            url = soup_next.get('href')
            soup = self.get_soup(url)
            soup_next = soup.find('a', class_=self.CLASS_NEXT)

            s_links = soup.find_all(
                'a', class_='Link ListingItemTitle-module__link')
            links += [link.get('href') for link in s_links]
            self.links = links

    def get_li_span2(self, soup, class_):
        try:
            text = soup.find('li', class_='CardInfoRow ' +
                             class_).find_all('span')[1].text
        except AttributeError:
            text = None
        return text

    def get_composition_dict(self, soup):
        result = []
        groups = soup.find_all('div', class_='CardComplectation__group')
        for group in groups:
            name = group.find(
                'div', class_='CardComplectation__item').get('data-group')
            values = group.find_all(
                'li', class_='CardComplectation__itemContentEl')
            values = [v.text for v in values]
            dict_gr = {'name': s2, 'values': s1}
            result.append(dict_gr)
        return result

    def read_doc_common(self, soup, df):
        data_attributes = soup.find('div', id='sale-data-attributes')
        data_attributes = json.loads(data_attributes.get(
            'data-bem'))['sale-data-attributes']

        df['bodyType'] = [
            soup.find('meta', itemprop='bodyType').get('content')]
        df['brand'] = [soup.find('meta', itemprop='brand').get('content')]
        df['color'] = [soup.find('meta', itemprop='color').get('content')]
        df['fuelType'] = [
            soup.find('meta', itemprop='fuelType').get('content')]
        df['modelDate'] = [
            soup.find('meta', itemprop='modelDate').get('content')]
        df['name'] = [soup.find('meta', itemprop='name').get('content')]
        df['numberOfDoors'] = [
            soup.find('meta', itemprop='numberOfDoors').get('content')]
        df['productionDate'] = [
            soup.find('meta', itemprop='productionDate').get('content')]
        df['vehicleConfiguration'] = [
            soup.find('meta', itemprop='vehicleConfiguration').get('content')]
        df['vehicleTransmission'] = [
            soup.find('meta', itemprop='vehicleTransmission').get('content')]
        df['engineDisplacement'] = [
            soup.find('meta', itemprop='engineDisplacement').get('content')]
        df['enginePower'] = [
            soup.find('meta', itemprop='enginePower').get('content')]
        df['description'] = [
            soup.find('meta', itemprop='description').get('content')]
        df['mileage'] = [data_attributes['km-age']]
        df['Комплектация'] = [self.get_composition_dict(soup)]
        df['Привод'] = [self.get_li_span2(soup, 'CardInfoRow_drive')]
        df['Руль'] = [self.get_li_span2(soup, 'CardInfoRow_wheel')]
        df['Состояние'] = [self.get_li_span2(soup, 'CardInfoRow_state')]
        df['Владельцы'] = [self.get_li_span2(soup, 'CardInfoRow_ownersCount')]
        df['ПТС'] = [self.get_li_span2(soup, 'CardInfoRow_pts')]
        df['Таможня'] = [self.get_li_span2(soup, 'CardInfoRow_customs')]
        df['Владение'] = [self.get_li_span2(soup, 'CardInfoRow_owningTime')]
        df['price'] = [data_attributes['price']]

        return df

    def read_doc_new(self, soup, df):
        df['Привод'] = [soup.find('li',
                                  class_='CardInfoGrouped__row CardInfoGrouped__row_drive'
                                  ).find('div', class_='CardInfoGrouped__cellValue').text]
        df['Руль'] = ['Левый']
        df['Состояние'] = ['Не требует ремонта']
        df['Владельцы'] = ['1 владелец']
        df['ПТС'] = ['Оригинал']
        df['Таможня'] = ['Растаможен']
        df['Владение'] = [None]

        return df

    def read_doc(self, link):
        soup = self.get_soup(link)
        df = pd.DataFrame([])

        try:
            df = self.read_doc_common(soup, df)
            if link[21:24] == 'new':
                df = self.read_doc_new(soup, df)
        except:
            print('sold ', link)
        return df

    def read_data(self):
        if self.links is None:
            self.read_links()

        data = map(self.read_doc, self.links[:])
        df = pd.DataFrame([])
        for d in tqdm(data):
            df = df.append(d, ignore_index=True)
        self.df = df

    def get_df(self):
        if self.df is None:
            self.read_data()
        return self.df

    def write_csv(self, path='out.csv'):
        df.to_csv(path, index=False, compression='gzip')
        print('=>', path)

    def write_xlsx(self, path='out.xlsx'):
        writer = pd.ExcelWriter(path)
        self.df.to_excel(writer, index=False, sheet_name='1')
        writer.save()
        print('=>', path)
