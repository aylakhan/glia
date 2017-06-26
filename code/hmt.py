#!/usr/bin/env python
#import numpy as np
import sys, shutil, socket, argparse
sys.path.append('/Users/ayla/devel/glia/code/gadget/python')
from script_util import *

dim = '2d'

t_resume = False
#t_resume = (len(sys.argv) > 1 and sys.argv[1] == '-r')

# paths
p_d = {
  'bin': '/Users/ayla/devel/glia/code/build',
  'dat': '/Users/ayla/devel/glia_data/ellisman',
  'res': '/Users/ayla/devel/glia_data/output_glia'}
p_dbin = p_d['bin'] + '/{b}'
p_fdat = {
  # ground truth segmentations
  'truth': p_d['dat'] + '/truth/region/' + dim + '/{s}/{i}.png',
  # raw images
  'gray': p_d['dat'] + '/im/gray/' + dim + '/{s}/{i}.mha',
  # boundary probability maps
  'pb': p_d['dat'] + '/im/chm/' + dim + '/{s}/{i}.mha',
  # blurred boundary probability maps
  'pbb': p_d['dat'] + '/im/chm-blur1/' + dim + '/{s}/{i}.mha',}
p_dres = {
  'segii': p_d['res'] + '/segii/{s}',  # watershed superpixels
  'segi': p_d['res'] + '/segi/{s}',  # pre-merged superpixels
  'order': p_d['res'] + '/order/{s}',  # merge trees
  'sal': p_d['res'] + '/sal/{s}',  # merge saliencies
  'bcf': p_d['res'] + '/bcf/{s}',  # boundary features
  'bcl': p_d['res'] + '/bcl/{s}',  # boundary labels
  'bcm': p_d['res'] + '/bcm/{s}',  # boundary classifier model
  'bcp': (p_d['res'] + '/bcp/{s}'),  # boundary predictions
  'seg': (p_d['res'] + '/seg/{s}'),  # final segmentations
}
p_fres = {}
for x in ['segii', 'segi', 'seg']: p_fres[x] = p_dres[x] + '/{i}.png'
for x in ['order', 'sal', 'bcf', 'bcl', 'bcp']:
  p_fres[x] = p_dres[x] + '/{i}.ssv'
p_fres['bcm'] = p_dres['bcm'] + '/bcm.bin'

# parameters
t_nproc = 70  # number of max parallel processes
t_nthrds= 1   # number of max parallel threads
t_wt = '0.02'  # initial water level
t_st = ['50', '200', '0.5']  # pre-merging params

# image ids
t_iids = {
  'tr': ['202', '204', '211', '217', '219', '224', '226', '235', '236',
         '239', '250', '251', '267', '269'],
  'te': ['205', '212', '214', '218', '221', '227', '232', '240', '242',
         '247', '256', '257', '259', '262',
         '203', '209', '213', '222', '223', '230', '238', '254', '255',
         '258', '263', '264', '266', '268',
         '200', '201', '210', '216', '220', '228', '231', '234', '241',
         '243', '246', '252', '260', '265',
         '206', '207', '208', '215', '225', '229', '233', '237', '244',
         '245', '248', '249', '253', '261']}
t_iids['all'] = t_iids['tr'] + t_iids['te']

def main ():
  # initial superpixels
  _jobs = list()
  make_dir(p_dres['segii'].format(s=''))
  for i in t_iids['all']:
    _f = p_fres['segii'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
        p_dbin.format(b='watershed'),
        '-i', p_fdat['pbb'].format(s='0', i=i),
        '-l', t_wt, '-u', 'true', '-o', _f]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=t_nthrds, name='segii')

  # pre-merging
  _jobs = list()
  make_dir(p_dres['segi'].format(s=''))
  for i in t_iids['all']:
    _f = p_fres['segi'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
        p_dbin.format(b='pre_merge'),
        '-s', p_fres['segii'].format(s='', i=i),
        '-p', p_fdat['pb'].format(s='0', i=i),
        '-t', t_st[0], t_st[1], '-b', t_st[2],
        '-r', 'true', '-u', 'true', '-o', _f]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=t_nthrds, name='segi')

  # trees and merging saliencies
  _jobs = list()
  make_dir(p_dres['order'].format(s=''))
  make_dir(p_dres['sal'].format(s=''))
  for i in t_iids['all']:
    _f_o = p_fres['order'].format(s='', i=i)
    _f_s = p_fres['sal'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f_o) and is_file_valid(_f_s)):
      _job = [
        p_dbin.format(b='merge_order_pb'),
        '-s', p_fres['segi'].format(s='', i=i),
        '-p', p_fdat['pb'].format(s='0', i=i),
        '-t', '1', '-o', _f_o, '-y', _f_s]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=t_nthrds, name='order/sal')

  # boundary features
  _jobs = list()
  make_dir(p_dres['bcf'].format(s=''))
  for i in t_iids['all']:
    _f = p_fres['bcf'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
        p_dbin.format(b='bc_feat'),
        '-s', p_fres['segi'].format(s='', i=i),
        '-o', p_fres['order'].format(s='', i=i),
        '-y', p_fres['sal'].format(s='', i=i),
        '--rbi', p_fdat['gray'].format(s='0', i=i),
        '--rbb', '16', '--rbl', '0.0', '--rbu', '1.0',
        '--rbi', p_fdat['pb'].format(s='0', i=i),
        '--rbb', '16', '--rbl', '0.0', '--rbu', '1.0',
        '--pb', p_fdat['pb'].format(s='0', i=i),
        '--s0', '1.0', '--sb', '1.0', '--bt', '0.2', '0.5', '0.8',
        '-n', 'false', '-l', 'false', '--simpf', 'false', '-b', _f]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=t_nthrds, name='bcf')

  # boundary labels
  _jobs = list()
  make_dir(p_dres['bcl'].format(s=''))
  for i in t_iids['all']:
    _f = p_fres['bcl'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
        p_dbin.format(b='bc_label_ri'),
        '-s', p_fres['segi'].format(s='', i=i),
        '-o', p_fres['order'].format(s='', i=i),
        '-t', p_fdat['truth'].format(s='0', i=i),
        '--f1', 'true', '-g', '0', '-p', 'false', '-w', 'false', '-l', _f]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=t_nthrds, name='bcl')

  # boundary classifier training
  _jobs = list()
  make_dir(p_dres['bcm']).format(s='')
  _f = p_fres['bcm'].format(s='')
  if not (t_resume and is_file_valid(_f)):
    _job = [
      p_dbin.format(b='train_rf'),
      '--nt', '255', '--mt', '0', '--sr', '0.7', '--ns', '1', '--bal', 'true',
      '--m', _f]
    for i in t_iids['tr']:
      _job.extend([
        '--f', p_fres['bcf'].format(s='', i=i),
        '--l', p_fres['bcl'].format(s='', i=i)])
    _jobs.append(_job)
  execute(_jobs, nproc=1, nt=1, name='bcm')

  # boundary predictions
  _jobs = list()
  make_dir(p_dres['bcp']).format(s='')
  for i in t_iids['all']:
    _f = p_fres['bcp'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
#        p_dbin.format(b='pred_tf'),
        p_dbin.format(b='pred_rf'),
        '--m', p_fres['bcm'].format(s=''), '--l', '-1',
        '--f', p_fres['bcf'].format(s='', i=i),
        '--p', p_fres['bcp'].format(s='', i=i)]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=1, name='bcp')

  # final segmentation
  _jobs = list()
  make_dir(p_dres['seg'].format(s=''))
  for i in t_iids['all']:
    _f = p_fres['seg'].format(s='', i=i)
    if not (t_resume and is_file_valid(_f)):
      _job = [
        p_dbin.format(b='segment_greedy'),
        '-s', p_fres['segi'].format(s='', i=i),
        '-o', p_fres['order'].format(s='', i=i),
        '-p', p_fres['bcp'].format(s='', i=i),
        '-r', 'true', '-u', 'true', '-f', _f]
      _jobs.append(_job)
  execute(_jobs, nproc=t_nproc, nt=1, name='seg')


if __name__ == "__main__": main()
