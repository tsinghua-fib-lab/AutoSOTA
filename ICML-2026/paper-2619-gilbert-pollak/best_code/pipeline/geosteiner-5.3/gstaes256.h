/***********************************************************************

	$Id: gstaes256.h,v 1.4 2022/11/19 13:45:52 warme Exp $

	File:	gstaes256.h
	Rev:	e-2
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Implementation of AES-256 cipher.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Created.
	e-2:	09/24/2016	warme
		: Reorganize include files.

************************************************************************/

#ifndef GSTAES256_H
#define	GSTAES256_H

#include "gsttypes.h"

/*
 * All information needed to encrypt and/or decrypt a data block.
 */

struct _gst_aes256_context {
	int8u		key [32];
	int8u		encryption_key [32];
	int8u		decryption_key [32];
};


extern void	_gst_aes256_clear (
				struct _gst_aes256_context * context);
extern void	_gst_aes256_decrypt_ecb (
				struct _gst_aes256_context *	context,
				int8u *				ciphertext);
extern void	_gst_aes256_encrypt_ecb (
				struct _gst_aes256_context *	context,
				int8u *				plaintext);
extern void	_gst_aes256_init (
				struct _gst_aes256_context *	context,
				const int8u *			key);

#endif
