/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'analysis_module.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <dlfcn.h>
#include <matrix.h>
#include <util.h>
#include <rng.h>

#include <analysis_module.h>
#include <analysis_table.h>

#define ANALYSIS_MODULE_TYPE_ID 6610123

struct analysis_module_struct {
  UTIL_TYPE_ID_DECLARATION;
  void                           * lib_handle;
  void                           * module_data;

  analysis_free_ftype            * freef;
  analysis_alloc_ftype           * alloc;
  analysis_initX_ftype           * initX;
  analysis_updateA_ftype         * updateA;
  analysis_init_update_ftype     * init_update;
  analysis_complete_update_ftype * complete_update;
  analysis_get_options_ftype     * get_options;
    
  analysis_set_int_ftype         * set_int;
  analysis_set_double_ftype      * set_double;
  analysis_set_bool_ftype        * set_bool; 
  analysis_set_string_ftype      * set_string;
  
  bool                             internal;  
  char                           * user_name;   /* String used to identify this module for the user; not used in 
                                                   the linking process. */
};



static analysis_module_type * analysis_module_alloc_empty( const char * user_name ) {
  analysis_module_type * module = util_malloc( sizeof * module , __func__ );
  UTIL_TYPE_ID_INIT( module , ANALYSIS_MODULE_TYPE_ID );

  module->lib_handle      = NULL;
  module->initX           = NULL;
  module->updateA         = NULL;
  module->set_int         = NULL;
  module->set_bool        = NULL;
  module->set_double      = NULL;
  module->set_string      = NULL;
  module->module_data     = NULL;
  module->init_update     = NULL;
  module->complete_update = NULL;
  module->user_name   = util_alloc_string_copy( user_name );

  return module;
}


static bool analysis_module_internal_check( analysis_module_type * module ) {
  return true;
}

static analysis_module_type * analysis_module_alloc__( rng_type * rng , 
                                                       const analysis_table_type * table , 
                                                       const char * user_name , 
                                                       void * lib_handle ) {

  analysis_module_type * module = analysis_module_alloc_empty( user_name );
  
  module->lib_handle        = lib_handle;
  module->initX             = table->initX;
  module->updateA           = table->updateA;
  module->init_update       = table->init_update;
  module->complete_update   = table->complete_update;
  module->set_int           = table->set_int;
  module->set_double        = table->set_double;
  module->set_string        = table->set_string;
  module->alloc             = table->alloc;
  module->freef             = table->freef;
  module->get_options       = table->get_options; 
  
  if (module->alloc != NULL)
    module->module_data = module->alloc( rng );

  if (!analysis_module_internal_check( module )) {
    fprintf(stderr,"** Warning loading module: %s failed - internal inconsistency\n", module->user_name);
    analysis_module_free( module );
    module = NULL;
  }

  return module;
}





static analysis_module_type * analysis_module_alloc( rng_type * rng , 
                                                     const char * libname , const char * user_name , 
                                                     const char * table_name ) {
  analysis_module_type * module = NULL;
  void * lib_handle = dlopen( libname , RTLD_NOW );
  if (lib_handle != NULL) {
    analysis_table_type * analysis_table = (analysis_table_type *) dlsym( lib_handle , table_name );
    if (analysis_table != NULL) {
      module = analysis_module_alloc__( rng , analysis_table , user_name , lib_handle );
    } else
      fprintf(stderr , "Failed to load symbol table:%s. Error:%s \n",table_name , dlerror());
    
    if (module == NULL) 
      dlclose( lib_handle );
  } else 
    fprintf(stderr , "Failed to load library:%s. Error:%s \n",libname , dlerror());
  
  if (module != NULL) {
    if (libname == NULL)
      module->internal = true;
    else
      module->internal = false;
  }

  return module;
}

analysis_module_type * analysis_module_alloc_internal( rng_type * rng , const char * user_name , const char * symbol_table ) {
  return analysis_module_alloc( rng , NULL , user_name , symbol_table );
}


analysis_module_type * analysis_module_alloc_external( rng_type * rng , const char * user_name , const char * lib_name) {
  return analysis_module_alloc( rng , lib_name , user_name , EXTERNAL_MODULE_TABLE );
}

const char * analysis_module_get_name( const analysis_module_type * module ) {
  return module->user_name;
}

/*****************************************************************/

static UTIL_SAFE_CAST_FUNCTION( analysis_module , ANALYSIS_MODULE_TYPE_ID )


void analysis_module_free( analysis_module_type * module ) {
  if (module->freef != NULL)
    module->freef( module->module_data );
  
  dlclose( module->lib_handle );
  free( module );
}


void analysis_module_free__( void * arg) {
  analysis_module_type * module = analysis_module_safe_cast( arg );
  analysis_module_free( module );
}

/*****************************************************************/
/* Update functions */


void analysis_module_initX(analysis_module_type * module , 
                           matrix_type * X , 
                           matrix_type * A , 
                           matrix_type * S , 
                           matrix_type * R , 
                           matrix_type * dObs , 
                           matrix_type * E , 
                           matrix_type * D ) {

  
  module->initX(module->module_data , X , A , S , R , dObs , E , D );
}


void analysis_module_updateA(analysis_module_type * module , 
                             matrix_type * A , 
                             matrix_type * S , 
                             matrix_type * R , 
                             matrix_type * dObs , 
                             matrix_type * E , 
                             matrix_type * D ) {

  module->updateA(module->module_data , A , S , R , dObs , E , D );
}




void analysis_module_init_update( analysis_module_type * module , 
                                  matrix_type * S , 
                                  matrix_type * R , 
                                  matrix_type * dObs , 
                                  matrix_type * E , 
                                  matrix_type * D ) {
  if (module->init_update != NULL)
    module->init_update( module->module_data , S , R , dObs , E , D);
}


void analysis_module_complete_update( analysis_module_type * module ) {
  if (module->complete_update != NULL)
    module->complete_update( module->module_data );
}



/*****************************************************************/


static bool analysis_module_set_int(analysis_module_type * module , const char * flag , int value) {
  if (module->set_int != NULL) 
    return module->set_int( module->module_data , flag , value );
  else
    return false;
}

static bool analysis_module_set_double(analysis_module_type * module , const char * var , double value) {
  if (module->set_double != NULL)
    return module->set_double( module->module_data , var , value );
  else
    return false;
}


static bool analysis_module_set_bool(analysis_module_type * module , const char * var , bool value) {
  if (module->set_bool != NULL)
    return module->set_bool( module->module_data , var , value );
  else
    return false;
}


static bool analysis_module_set_string(analysis_module_type * module , const char * var , const char * value) {
  if (module->set_string != NULL)
    return module->set_string( module->module_data , var , value );
  else
    return false;
}



/* 
   The input value typically comes from the configuration system and
   is in terms of a string, irrespective of the fundamental type of
   the underlying parameter. The algorithm for setting the parameter
   tries datatypes as follows: integer - double - string.

   For the numeric datatypes the algorithm is two step:

     1. Try the conversion string -> numeric.
     2. Try calling the analysis_module_set_xxx() function.

   Observe that this implies that the same variable name can NOT be
   used for different variable types.
*/

bool analysis_module_set_var( analysis_module_type * module , const char * var_name , const char * string_value ) {
  bool set_ok = false;
  {
    int  int_value;
    
    if (util_sscanf_int( string_value , &int_value )) 
      set_ok = analysis_module_set_int( module , var_name , int_value );
    
    if (set_ok)
      return true;
  }
  
  {
    double double_value;
    if (util_sscanf_double( string_value , &double_value )) 
      set_ok = analysis_module_set_double( module , var_name , double_value );

    if (set_ok)
      return true;
  }

  {
    bool bool_value;
    if (util_sscanf_bool( string_value , &bool_value))
      set_ok = analysis_module_set_bool( module , var_name , bool_value );
  }
  
  

  set_ok = analysis_module_set_string( module , var_name , string_value );
  if (!set_ok)
    fprintf(stderr,"** Warning: failed to set %s=%s for analysis module:%s\n", var_name , string_value , module->user_name);
  
  return set_ok;
}


bool analysis_module_get_option( const analysis_module_type * module , long flag) {
  return (flag & module->get_options( module->module_data , flag ));
}
